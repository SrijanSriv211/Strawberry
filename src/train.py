from utils import calc_total_time, print_banner, print0
from model import Strawberry, Config
from encoder import Encoder
from optimizer import Muon

from colorama import Style, Fore, init
from rich.progress import track
from itertools import chain

import random, pickle, torch, json, time, math, sys, os

torch._inductor.config.coordinate_descent_tuning = True
torch._dynamo.config.compiled_autograd = True
init(autoreset=True)

# load config
CONFIG = json.loads(open(sys.argv[1], "r", encoding="utf-8").read()) if len(sys.argv) > 1 else {
	"dataset": {
		"data_division": 0.8,
		"load_from_file": True,
		"path": "data/webtext.bin"
	},
	"checkpoints": {
		"path": "bin/ck",
		"interval": 2000,
		"create_checkpoints": True
	},
	"model_hyperparams": {
		"vocab_size": 8192,
		"block_size": 256,
		"n_layer": 2,
		"r_layer": 2,
		"n_head": 4,
		"n_embd": 64,
		"n_qkv": 256,
		"n_ffn": 256
	},
	"optimizer_hyperparams": {
		"eps": 1e-10,
		"beta1": 0.9,
		"beta2": 0.95,
		"weight_decay": 1e-1,
		"use_muon": False,
		"momentum": 0.95
	},
	"model_path": "bin/air.strawberry",
	"encoder_path": "bin/cl8k.bin",
	"init_from": "scratch",
	"seed": "auto",

	"gradient_accumulation_steps": 1,
	"batch_size": 4,

	"max_iters": 50000,
	"eval_interval": 2000,
	"log_interval": 200,
	"eval_iters": 200,

	"decay_lr": True,
	"lr_decay_iters": 50000,
	"learning_rate": 3e-3,
	"cooldown_frac": 0.4,
	"warmup_iters": 2000,
	"min_lr": 3e-4
}

# init seed
if CONFIG["seed"] != "auto":
	torch.manual_seed(CONFIG["seed"])
	random.seed(CONFIG["seed"])

if CONFIG["checkpoints"]["create_checkpoints"] and not os.path.isdir(CONFIG["checkpoints"]["path"]):
	os.mkdir(CONFIG["checkpoints"]["path"])

log_path = CONFIG["checkpoints"]["path"] if CONFIG["checkpoints"]["create_checkpoints"] else "bin"

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
init_from = CONFIG["init_from"][11:] if CONFIG["init_from"].startswith("pretrained,") else "scratch"

# print the device
print_banner()
print0(f"config: {Fore.WHITE}{Style.DIM}`{json.dumps(CONFIG)}`", overwrite=(init_from == "scratch"), log_path=log_path)
print0("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{device}", log_path=log_path)

# load stats
checkpoint = None if init_from == "scratch" else torch.load(init_from)
stats = checkpoint["stats"] if checkpoint is not None and "stats" in checkpoint.keys() else {
	"step": 0,
	"loss": {
		"train": [],
		"test": [],
		"val": []
	},
	"lr": []
}

# create an instance of Strawberry
hyperparams = CONFIG["model_hyperparams"] if checkpoint is None else checkpoint["hyperparams"]
conf = Config(**hyperparams)
model = Strawberry(conf)

# load the state dict
if checkpoint is not None:
	model.load_state_dict(checkpoint["model"])
model.to(device)

# optimizers!
optimizer_hyperparams = CONFIG["optimizer_hyperparams"] if checkpoint is None else checkpoint["optimizer_hyperparams"]

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
adam_params = embed_params

if not optimizer_hyperparams["use_muon"]:
	adam_params = embed_params + hidden_matrix_params

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.AdamW(
	adam_params, lr=CONFIG["learning_rate"], betas=(optimizer_hyperparams["beta1"], optimizer_hyperparams["beta2"]),
	eps=optimizer_hyperparams["eps"], weight_decay=optimizer_hyperparams["weight_decay"], fused=True
)
optimizers = [optimizer1]

if optimizer_hyperparams["use_muon"]:
	optimizer2 = Muon(
		hidden_matrix_params,
		lr=CONFIG["learning_rate"],
		momentum=optimizer_hyperparams["momentum"],
		weight_decay=optimizer_hyperparams["weight_decay"]
	)
	optimizers.append(optimizer2)

# load optimizer(s) state dict if loading from checkpoint
if checkpoint is not None:
	for o, s in zip(optimizers, checkpoint["optimizers"]):
		o.load_state_dict(s)

class dataloader:
	def __init__(self, path, block_size, batch_size, sink_tok, data_division=0.8, isfile=True):
		self.path = path
		self.data_division = data_division
		self.block_size, self.batch_size = block_size, batch_size

		self.files = [path] if isfile else [os.path.join(path, i) for i in os.listdir(path)]
		self.sink_col = torch.full((self.batch_size, 1), sink_tok)

	def load_dataset(self):
		self.train, self.val = [], []

		for file in self.files:
			with open(file, "rb") as f:
				dataset = pickle.load(f)["dataset"]

			random.shuffle(dataset)
			flat_dataset = chain.from_iterable(dataset)
			del dataset

			flat_dataset = list(flat_dataset)
			n_train_toks = int(len(flat_dataset) * self.data_division)
			n_val_toks = len(flat_dataset) - n_train_toks

			self.train.extend(flat_dataset[:n_train_toks])
			self.val.extend(flat_dataset[n_train_toks:])

		self.train = torch.tensor(self.train, dtype=torch.int64)
		self.val = torch.tensor(self.val, dtype=torch.int64)
		return n_train_toks, n_val_toks

	def next_batch(self, split):
		data = self.train if split == "train" else self.val
		ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
		y = torch.stack([data[i:i + self.block_size] for i in ix])
		x = torch.cat([self.sink_col, y[:, :-1]], dim=1) # x: prepend SINK, drop last token
		return x.to(device), y.to(device)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, next_batch):
	out = {}
	model.eval()
	for split in ["train", "val"]:
		losses = torch.zeros(CONFIG["eval_iters"])
		for k in track(range(CONFIG["eval_iters"]), description=f"{Fore.WHITE}{Style.BRIGHT}calc {Fore.WHITE}{Style.DIM}{split} loss{Style.RESET_ALL}"):
			X, Y = next_batch(split)
			_, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

def get_state(model: Strawberry, optimizers: list[torch.optim.AdamW, Muon]):
	state_dict = model.state_dict()
	unwanted_prefix = '_orig_mod.'

	for k, v in list(state_dict.items()):
		if k.startswith(unwanted_prefix):
			state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

	return {
		"stats": stats,
		"device": device,
		"model": state_dict,
		"hyperparams": hyperparams,
		"optimizer_hyperparams": optimizer_hyperparams,
		"optimizers": [o.state_dict() for o in optimizers]
	}

# load encoder
enc = Encoder()
enc.load(CONFIG["encoder_path"])
sink_tok = enc.special_tokens["<|sink|>"]

# load dataset
dataset = dataloader(
	CONFIG["dataset"]["path"],
	hyperparams["block_size"], CONFIG["batch_size"], sink_tok,
	CONFIG["dataset"]["data_division"], CONFIG["dataset"]["load_from_file"]
)
n_train_toks, n_val_toks = dataset.load_dataset()

print0(f"{Fore.WHITE}{Style.BRIGHT}{((n_train_toks + n_val_toks)/1e6)}M", "total tokens", log_path=log_path)
print0(
	f"{Fore.WHITE}{Style.BRIGHT}{(n_train_toks/1e6)}M", "train tokens,",
	f"{Fore.WHITE}{Style.BRIGHT}{(n_val_toks/1e6)}M", "val tokens",
	log_path=log_path
)

# report number of parameters
print0(
	f"{Fore.WHITE}{Style.BRIGHT}{sum(p.numel() for p in model.parameters())/1e6}M", "parameters,",
	f"{Fore.WHITE}{Style.BRIGHT}{sum(p.numel() for p in model.blocks.parameters())/1e6}M", "non-embedding parameters",
	log_path=log_path
)

# compile the model
print0(f"compiling the model... {Fore.WHITE}{Style.DIM}(takes a ~minute)", log_path=log_path)
model = torch.compile(model)

# training loop
# start training the model
print0("started training", log_path=log_path)
start_time = eval_t0 = test_t0 = time.time()
n_steps = CONFIG["max_iters"] - stats["step"]
steps_per_epoch = int((n_train_toks + n_val_toks) / (hyperparams["block_size"] * CONFIG["batch_size"]))

for _ in range(n_steps):
	# determine and set the learning rate for this iteration
	## learning rate decay scheduler (cosine with warmup)
	if not CONFIG["decay_lr"]:
		lr = CONFIG["learning_rate"]

	## 1) linear warmup for warmup_iters steps
	elif stats["step"] < CONFIG["warmup_iters"]:
		lr = CONFIG["learning_rate"] * (stats["step"] + 1) / (CONFIG["warmup_iters"] + 1)

	## 2) constant learning rate for some time
	elif stats["step"] / CONFIG["lr_decay_iters"] <= 1 - CONFIG["cooldown_frac"]:
		lr = CONFIG["learning_rate"]

	## 3) if stats["step"] > lr_decay_iters, lr = min learning rate
	elif stats["step"] > CONFIG["lr_decay_iters"]:
		lr = CONFIG["min_lr"]

	## 4) in between, use cosine decay down to min learning rate
	else:
		const_lr_iters = int((1 - CONFIG["cooldown_frac"]) * CONFIG["lr_decay_iters"])
		decay_ratio = (stats["step"] - const_lr_iters) / (CONFIG["lr_decay_iters"] - const_lr_iters)

		assert 0 <= decay_ratio <= 1
		coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
		lr = CONFIG["min_lr"] + coeff * (CONFIG["learning_rate"] - CONFIG["min_lr"])

	## set optimizers' learning rate
	for o in optimizers:
		for group in o.param_groups:
			group["lr"] = lr
	stats["lr"].append(lr)

	# training section
	for _ in range(CONFIG["gradient_accumulation_steps"]):
		X, Y = dataset.next_batch("train")
		_, loss = model(X, Y)

		# scale the loss to account for gradient accumulation
		loss = loss / CONFIG["gradient_accumulation_steps"]

		# backward pass
		loss.backward()

	if optimizer_hyperparams["use_muon"]:
		for group in optimizers[1].param_groups:
			frac = min(stats["step"] / 300, 1) # momentum warmup for muon
			group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

	## step the optimizers
	for o in optimizers:
		o.step()

	## flush the gradients as soon as we can, no need for this memory anymore
	optimizers[0].zero_grad(set_to_none=True)
	model.zero_grad(set_to_none=True)

	# validation section
	## save checkpoint
	if CONFIG["checkpoints"]["create_checkpoints"] and stats["step"] > 0 and stats["step"] % CONFIG["checkpoints"]["interval"] == 0:
		print0(f"saved checkpoint at step {Fore.WHITE}{Style.BRIGHT}{stats["step"]}", log_path=log_path)
		torch.save(get_state(model, optimizers), f"{CONFIG["checkpoints"]["path"]}/step{stats["step"]}.strawberry")

	## log train-val loss
	if stats["step"] > 0 and stats["step"] % CONFIG["eval_interval"] == 0:
		losses = estimate_loss(model, dataset.next_batch)
		eval_t1 = time.time()
		eval_dt = eval_t1 - eval_t0
		eval_t0 = eval_t1

		print0(
			f"{Fore.WHITE}{Style.BRIGHT}step",
			f"{Fore.WHITE}{Style.DIM}[{stats["step"]}/{CONFIG["max_iters"]}]"
			f"{Fore.RESET}{Style.RESET_ALL}:",
			f"train loss {Fore.WHITE}{Style.BRIGHT}{losses["train"]:.4f}"
			f"{Fore.RESET}{Style.RESET_ALL},",
			f"val loss {Fore.WHITE}{Style.BRIGHT}{losses["val"]:.4f}"
			f"{Fore.RESET}{Style.RESET_ALL},",
			f"lr {Fore.WHITE}{Style.BRIGHT}{lr:.7f}"
			f"{Fore.RESET}{Style.RESET_ALL},",
			f"time took {Fore.WHITE}{Style.DIM}{calc_total_time(eval_dt)}",
			log_path=log_path
		)
		stats["loss"]["train"].append(losses["train"])
		stats["loss"]["val"].append(losses["val"])

		### sample generation
		out = model.generate([], sink_tok, hyperparams["block_size"])[0].tolist()
		print0(f"{Fore.WHITE}{Style.DIM}```\n{enc.decode(out)}\n```", log_path=log_path)

	## log test loss
	### get loss as float. note: this is a CPU-GPU sync point
	### scale up to undo the division above, approximating the true total loss (exact would have been a sum)
	lossf = loss.item() * CONFIG["gradient_accumulation_steps"]
	stats["loss"]["test"].append(lossf)

	if stats["step"] % CONFIG["log_interval"] == 0:
		test_t1 = time.time()
		test_dt = test_t1 - test_t0
		test_t0 = test_t1

		toks_per_sec = (CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"] * hyperparams["block_size"] * CONFIG["log_interval"]) / test_dt
		print0(
			f"{Fore.WHITE}{Style.BRIGHT}iter",
			f"{Fore.WHITE}{Style.DIM}[{stats["step"]}/{CONFIG["max_iters"]}]"
			f"{Fore.RESET}{Style.RESET_ALL}:",
			f"loss {Fore.WHITE}{Style.BRIGHT}{lossf:.4f}"
			f"{Fore.RESET}{Style.RESET_ALL},",
			f"dt {Fore.WHITE}{Style.DIM}{calc_total_time(test_dt)}"
			f"{Fore.RESET}{Style.RESET_ALL},",
			f"tok/s {Fore.WHITE}{Style.DIM}{toks_per_sec:.2f}",
			log_path=log_path
		)
	stats["step"] += 1

print0("total time:", calc_total_time(time.time() - start_time), log_path=log_path)
torch.save(get_state(model, optimizers), CONFIG["model_path"])
