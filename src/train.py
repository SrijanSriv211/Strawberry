from model import Config, Strawberry
from sample import generate
from optimizer import Muon
from colorama import Style, Fore, init
from rich.progress import track
import random, torch, regex, numpy, json, time, math, sys, os

# initialization
init(autoreset=True)
torch._inductor.config.coordinate_descent_tuning = True
torch._dynamo.config.compiled_autograd = True

# load config
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "script/config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
	CONFIG = json.load(f)

# set device
device = ("cuda" if torch.cuda.is_available() else "cpu") if CONFIG["device"] == "auto" else CONFIG["device"]

# init seed
if CONFIG["seed"] != "auto":
	torch.manual_seed(CONFIG["seed"])
	numpy.random.seed(CONFIG["seed"])
	random.seed(CONFIG["seed"])

# save the text in a text file
ansi_escape = regex.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
def print0(*text, println=True, overwrite=False, save_to_file=True):
	if println:
		print(*text)

	if not save_to_file:
		return

	# save cleaned text to the file
	if not os.path.isdir(CONFIG["checkpoints"]["path"]):
		os.mkdir(CONFIG["checkpoints"]["path"])

	with open(os.path.join(CONFIG["checkpoints"]["path"], "out.txt"), "w" if overwrite else "a", encoding="utf-8") as f:
		f.write(" ".join(tuple(ansi_escape.sub('', part) for part in text)) + "\n")

def calc_total_time(seconds):
    # separate the integer part (for hours, minutes, and seconds) from the fractional part (for milliseconds)
    sec_int, millis = divmod(seconds, 1)
    millis = int(millis * 1000) # convert the fractional part to milliseconds

    min, sec = divmod(int(sec_int), 60)
    hour, min = divmod(min, 60)
    hours, minutes, seconds = int(hour), int(min), int(sec)

    t = [
        f"{hours} hour" + ("s" if hours > 1 else "") if hours > 0 else None,
        f"{minutes} minute" + ("s" if minutes > 1 else "") if minutes > 0 else None,
        f"{seconds} second" + ("s" if seconds > 1 else "") if seconds > 0 else None,
        f"{millis} ms" if millis > 0 else None
    ]
    t = list(filter(None, t))

    return ", ".join(t) if t else "0 seconds"

def init_model(checkpoint=None):
	# print the device
	print0(f"```config.json\n{json.dumps(CONFIG, indent=4)}\n```", println=False, overwrite=True if checkpoint is None else False)
	print0("Training on", f"{Fore.YELLOW}{Style.BRIGHT}{device}")

	# load stats
	stats = checkpoint["stats"] if checkpoint is not None and "stats" in checkpoint.keys() else {
		"steps": 0,
		"train": [],
		"eval": [],
		"val": [],
		"lr": []
	}

	# load hyperparams
	hyperparams = dict()
	# read off the created CONFIG params, so we can store them into checkpoint correctly
	for k in ["vocab_size", "block_size", "n_layer", "d_layer", "n_head", "n_embd", "d_rank", "d_qkv"]:
		hyperparams[k] = CONFIG[k]

	# create an instance of Strawberry
	conf = Config(**hyperparams)
	model = Strawberry(conf)
	# load the state dict
	if checkpoint is not None:
		model.load_state_dict(checkpoint["model"])
	model.to(device)

	return model, hyperparams, stats

# optimizers!
def configure_optimizers(model: Strawberry, checkpoint=None):
	# collect the parameters to optimize
	hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
	embed_params = [p for n, p in model.named_parameters() if "embed" in n]
	adam_params = embed_params

	if not CONFIG["use_muon"]:
		adam_params = adam_params + hidden_matrix_params

	# init the optimizer(s)
	# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
	# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
	optimizer1 = torch.optim.AdamW(
		adam_params, lr=CONFIG["learning_rate"], betas=(CONFIG["beta1"], CONFIG["beta2"]),
		eps=CONFIG["eps"], weight_decay=CONFIG["weight_decay"], fused=True
	)
	optimizers = [optimizer1]

	if CONFIG["use_muon"]:
		optimizer2 = Muon(hidden_matrix_params, lr=CONFIG["learning_rate"], momentum=CONFIG["momentum"], weight_decay=CONFIG["weight_decay"])
		optimizers.append(optimizer2)

	# load optimizer(s) state dict if loading from checkpoint
	if checkpoint is not None:
		for o, s in zip(optimizers, checkpoint["optimizers"]):
			o.load_state_dict(s)
	return optimizers

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
	if not CONFIG["decay_lr"]:
		return CONFIG["learning_rate"]

	# 1) linear warmup for warmup_iters steps
	elif it < CONFIG["warmup_iters"]:
		return CONFIG["learning_rate"] * (it + 1) / (CONFIG["warmup_iters"] + 1)

	# 2) constant learning rate for some time
	elif it / CONFIG["lr_decay_iters"] <= 1 - CONFIG["cooldown_frac"]:
		return CONFIG["learning_rate"]

	# 3) if it > lr_decay_iters, return min learning rate
	elif it > CONFIG["lr_decay_iters"]:
		return CONFIG["min_lr"]

	# 4) in between, use cosine decay down to min learning rate
	const_lr_iters = int((1 - CONFIG["cooldown_frac"]) * CONFIG["lr_decay_iters"])
	decay_ratio = (it - const_lr_iters) / (CONFIG["lr_decay_iters"] - const_lr_iters)

	assert 0 <= decay_ratio <= 1
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
	return CONFIG["min_lr"] + coeff * (CONFIG["learning_rate"] - CONFIG["min_lr"])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, get_batch):
	out = {}
	model.eval()
	for split in ["train", "val"]:
		losses = torch.zeros(CONFIG["eval_iters"])
		for k in track(range(CONFIG["eval_iters"]), description=f"{Fore.WHITE}{Style.BRIGHT}calc {Fore.WHITE}{Style.DIM}{split} loss{Style.RESET_ALL}"):
			X, Y = get_batch(split)
			_, loss = model(X, Y)

			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

class dataloader:
	def __init__(self, path, isfile=True, t_in_mem=50_000_000, exhaust_pool=True):
		self.path = path
		self.files = [path] if isfile else [os.path.join(path, i) for i in os.listdir(path)]
		self.orig_files = self.files[:]
		self.t_in_mem = t_in_mem # tokens in memory
		self.exhaust_pool = exhaust_pool

	# get total number of tokens
	def get_tok_count(self, orig=True):
		files = self.orig_files if orig else self.files
		return sum([numpy.memmap(f, dtype=numpy.int16, mode="r").size for f in files])

	def load_dataset(self):
		if len(self.files) <= 0 or self.get_tok_count(False) < self.t_in_mem:
			self.files = self.orig_files[:]

		self.data = []
		for f in random.sample(self.files, k=len(self.files)):
			self.data.extend(numpy.memmap(f, dtype=numpy.int16, mode="r"))
			self.files.remove(f) # remove file until the next epoch

			if self.t_in_mem is not None and round(len(self.data) / self.t_in_mem, 1) >= 0.8:
				break

		block_size = CONFIG["block_size"] + 1
		self.data = self.data[:len(self.data) // block_size * block_size]
		self.data = numpy.array(self.data, dtype=numpy.int16).reshape(-1, block_size)
		self.data = torch.from_numpy(self.data.astype(numpy.int64))
		self.batches = torch.randperm(self.data.shape[0]) if self.exhaust_pool else None
		self.ptr = 0

	def next_batch(self):
		if self.ptr + CONFIG["batch_size"] > self.data.shape[0]:
			self.load_dataset()

		# get x, y batches
		# sample data without replacement during training (until an epoch boundary is reached) is minimize overfitting.
		if self.exhaust_pool:
			ix = self.batches[self.ptr:self.ptr + CONFIG["batch_size"]]
			self.ptr += CONFIG["batch_size"]

		else:
			ix = torch.randint(self.data.shape[0], (CONFIG["batch_size"],))

		data = self.data[ix]
		x = data[:, :CONFIG["block_size"]].contiguous()
		y = data[:, 1:1+CONFIG["block_size"]].contiguous()
		x, y = x.to(device), y.to(device)
		return x, y

# init model and optimizers
init_from = torch.load(CONFIG["init_from"][11:]) if CONFIG["init_from"].startswith("pretrained,") else None
model, hyperparams, stats = init_model(init_from)
optimizers = configure_optimizers(model, init_from)

# load train and val data
train_data_loader = dataloader(CONFIG["train_data"], CONFIG["load_from_file"])
val_data_loader = dataloader(CONFIG["val_data"], CONFIG["load_from_file"], exhaust_pool=False)
train_data_loader.load_dataset()
val_data_loader.load_dataset()
# simple lambda function for `estimate_loss` function
get_batch = lambda x: train_data_loader.next_batch() if x == "train" else val_data_loader.next_batch()

# print the number of tokens
num_train_toks = train_data_loader.get_tok_count()
num_val_toks = val_data_loader.get_tok_count()
print0(f"{Fore.WHITE}{Style.BRIGHT}{((num_train_toks + num_val_toks)/1e6)}M", "total tokens")
print0(
	f"{Fore.WHITE}{Style.BRIGHT}{(num_train_toks/1e6)}M", "train tokens,", f"{Fore.WHITE}{Style.BRIGHT}{(num_val_toks/1e6)}M", "val tokens",
	f"   {Fore.WHITE}{Style.DIM}(using train tokens as val tokens)" if CONFIG["train_data"] == CONFIG["val_data"] else ""
)
del num_train_toks, num_val_toks

# report number of parameters
print0(f"{Fore.WHITE}{Style.BRIGHT}{sum(p.numel() for p in model.parameters())/1e6}M", "parameters")

# compile the model
if CONFIG["compile"]:
	print0(f"compiling the model... {Fore.WHITE}{Style.DIM}(takes a ~minute)")
	model = torch.compile(model) # requires PyTorch 2.0

# training loop
start_time = eval_t0 = t0 = time.time()

def get_trained_model(model: Strawberry, optimizers: list[torch.optim.AdamW, Muon]):
	state_dict = model.state_dict()
	unwanted_prefix = '_orig_mod.'
	for k, v in list(state_dict.items()):
		if k.startswith(unwanted_prefix):
			state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

	return {
		"model": state_dict,
		"optimizers": [o.state_dict() for o in optimizers],
		"hyperparams": hyperparams,
		"device": device,
		"stats": stats
	}

# write checkpoints
def save_checkpoint(model, optimizer):
	if CONFIG["checkpoints"] == None or stats["steps"] <= 0 or stats["steps"] % CONFIG["checkpoints"]["interval"] != 0: return
	if not os.path.isdir(CONFIG["checkpoints"]["path"]): os.mkdir(CONFIG["checkpoints"]["path"])
	print0(f"saved checkpoint at step {Fore.WHITE}{Style.BRIGHT}{stats["steps"]}")
	torch.save(get_trained_model(model, optimizer), f"{CONFIG["checkpoints"]["path"]}/s{stats["steps"]}.bin")

# generate some sample text
def sample_output(model, optimizer):
	if CONFIG["sample_interval"] == None or stats["steps"] <= 0 or stats["steps"] % CONFIG["sample_interval"] != 0: return
	out = generate(get_trained_model(model, optimizer), CONFIG["encoder_path"], l=CONFIG["block_size"], T=[None])
	print0(f"{Fore.WHITE}{Style.DIM}```s{stats["steps"]}.bin\n{out}\n```")

# evaluate the loss on train/val sets
def log_eval_loss():
	if stats["steps"] <= 0 or stats["steps"] % CONFIG["eval_interval"] != 0:
		return
	global eval_t0

	# timing and logging
	losses = estimate_loss(model, get_batch)
	eval_t1 = time.time()
	eval_dt = eval_t1 - eval_t0
	eval_t0 = eval_t1
	print0(
		f"{Fore.WHITE}{Style.BRIGHT}step",
		f"{Fore.WHITE}{Style.DIM}[{stats["steps"]}/{CONFIG["max_iters"]}]"
		f"{Fore.RESET}{Style.RESET_ALL}:",
		f"train loss {Fore.WHITE}{Style.BRIGHT}{losses["train"]:.4f}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"val loss {Fore.WHITE}{Style.BRIGHT}{losses["val"]:.4f}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"lr {Fore.WHITE}{Style.BRIGHT}{lr:.7f}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"time took {Fore.WHITE}{Style.DIM}{calc_total_time(eval_dt)}"
	)
	stats["train"].append(losses["train"])
	stats["val"].append(losses["val"])

def log_loss():
	if stats["steps"] % CONFIG["log_interval"] != 0:
		return
	global t0

	# timing and logging
	t1 = time.time()
	dt = t1 - t0
	t0 = t1

	# get loss as float. note: this is a CPU-GPU sync point
	# scale up to undo the division above, approximating the true total loss (exact would have been a sum)
	lossf = loss.item() * CONFIG["gradient_accumulation_steps"]

	toks_per_sec = (CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"] * CONFIG["block_size"] * CONFIG["log_interval"]) / dt
	print0(
		f"{Fore.WHITE}{Style.BRIGHT}iter",
		f"{Fore.WHITE}{Style.DIM}[{stats["steps"]}/{CONFIG["max_iters"]}]"
		f"{Fore.RESET}{Style.RESET_ALL}:",
		f"loss {Fore.WHITE}{Style.BRIGHT}{lossf:.4f}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"dt {Fore.WHITE}{Style.DIM}{calc_total_time(dt)}"
		f"{Fore.RESET}{Style.RESET_ALL},",
		f"tok/s {Fore.WHITE}{Style.DIM}{toks_per_sec:.2f}"
	)
	stats["eval"].append(lossf)

# forward backward update, with optional gradient accumulation to simulate larger batch size
# and using the GradScaler if data type is float16
def train_model():
	global optimizers, model, get_batch
	for _ in range(CONFIG["gradient_accumulation_steps"]):
		# immediately async prefetch next batch while model is doing the forward pass on the GPU
		X, Y = get_batch("train")

		_, loss = model(X, Y)
		loss = loss / CONFIG["gradient_accumulation_steps"] # scale the loss to account for gradient accumulation

		# backward pass, with gradient scaling if training in fp16
		loss.backward()

	# clip the gradient
	if CONFIG["grad_clip"] != 0.0:
		torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

	if CONFIG["use_muon"]:
		for group in optimizers[1].param_groups:
			frac = min(stats["steps"] / 300, 1) # momentum warmup for muon
			group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

	# step the optimizers and scaler if training in fp16
	for o in optimizers:
		o.step()

	# flush the gradients as soon as we can, no need for this memory anymore
	optimizers[0].zero_grad(set_to_none=True)
	model.zero_grad(set_to_none=True)
	return loss

# warmup the training kernels
print0(f"warming up training kernels... {Fore.WHITE}{Style.DIM}(takes a ~minute)")
init_from_pretrained = CONFIG["init_from"].startswith("pretrained,") and stats["steps"] > 0
for i in range(stats["steps"] if init_from_pretrained else 10):
	if not init_from_pretrained:
		train_model()
		continue

	if i >= 0 and i % CONFIG["eval_interval"] == 0:
		[get_batch(split) for split in ["train", "val"] for _ in range(CONFIG["eval_iters"])]
	get_batch("train")

# start training the model
print0("started training")
n_steps = CONFIG["max_iters"] - stats["steps"]
for _ in range(n_steps):
	# determine and set the learning rate for this iteration
	lr = get_lr(stats["steps"])
	for o in optimizers:
		for group in o.param_groups:
			group["lr"] = lr
	stats["lr"].append(lr)

	# validation section
	# save checkpoint and log sample and eval loss
	save_checkpoint(model, optimizers)
	sample_output(model, optimizers)
	log_eval_loss()

	# training section
	loss = train_model()

	# logging
	log_loss()
	stats["steps"] += 1

print0("total time:", calc_total_time(time.time() - start_time))
torch.save(get_trained_model(model, optimizers), CONFIG["save_path"])
