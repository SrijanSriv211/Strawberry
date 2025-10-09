from encoder import Encoder
from rich.progress import track
from colorama import Style, Fore, init
import itertools, argparse, numpy, json, os

init(autoreset=True)

parser = argparse.ArgumentParser(description="A powerful text encryption and decryption program.")
parser.add_argument("-i", help="dataset path", required=True)
parser.add_argument("-o", help="output path", required=True)
parser.add_argument("-e", help="encoder path", required=True)
parser.add_argument("-d", help="train-val data division ratio", type=float, default=0.9)
parser.add_argument("-s", help="max toks in each data shard", type=int, default=50_000_000)
args = parser.parse_args()

CONFIG = {
	"dataset_path": args.i,
	"outpath": args.o,
	"enc_path": args.e,
	"data_division": args.d,
	"toks_per_shard": args.s
}

"""
Load encoder
"""
enc = Encoder()
enc.load(CONFIG["enc_path"])
data = []

"""
Pretraining dataset
"""
dataset_files = [os.path.join(CONFIG["dataset_path"], i) for i in os.listdir(CONFIG["dataset_path"])]
dataset_files = sorted(dataset_files, key=os.path.getsize)

# python src/prepare_data.py -i data/base/json -o data/base -e bin/cl8k.bin
total_unique_chars, total_chars = 0, 0
total_train_chars, total_val_chars = 0, 0
total_train_tokens, total_val_tokens = 0, 0
lensum = lambda x: sum([len(i) for i in x])
id = 1

def save_data(data, split):
	global id
	if not os.path.isdir(os.path.join(CONFIG["outpath"], split)):
		os.mkdir(os.path.join(CONFIG["outpath"], split))

	jump = CONFIG["toks_per_shard"]
	num_shards = lensum(data) // CONFIG["toks_per_shard"] + 1
	data = list(itertools.chain.from_iterable(data))
	for i in range(num_shards):
		outpath = f"{CONFIG["outpath"]}/{split}/{split}" + (f"_{id}" if num_shards > 1 else "") + ".bin"
		id += 1

		print(outpath)
		numpy.array(data[i * jump : (i+1) * jump], dtype=numpy.int16).tofile(outpath)

def encode_data(data, split):
	num_chars = lensum(data)

	for i, x in enumerate(track(data, f"{Fore.WHITE}{Style.BRIGHT}encoding {Fore.WHITE}{Style.DIM}{split} chars{Style.RESET_ALL}")):
		data[i] = enc.encode(f"<|sot|>{x}<|eot|>\n", allowed_special="all")
		if i == 0:
			print("======= SAMPLE TEXT FROM DATASET =======")
			print(f"{Fore.WHITE}{Style.DIM}```\n<|sot|>{x}<|eot|>\n```", "\n\n")
			print(data[i], "\n\n")
			print(f"{Fore.WHITE}{Style.DIM}```\n{enc.decode(data[i])}```", "\n\n")

	print(f"{(num_chars/1e6)}M {split} chars,", f"{(lensum(data)/1e6)}M {split} tokens")
	return num_chars

# split train and val data based on `data_division`
def split_data(train_data):
	val_size = int(num_chars * (1 - CONFIG["data_division"]))
	val_data, idx, size = [], [], 0
	for i, x in enumerate(train_data):
		if size >= val_size:
			break

		elif i % (CONFIG["data_division"] * 10) != 0:
			continue

		val_data.append(x)
		idx.append(i)
		size += len(x)

	# remove items from `train_data` which are now a part of `val_data`
	idx.sort(reverse=True)
	[train_data.pop(i) for i in idx]
	return val_data

train_data = []
num_chars_to_enc_at_once = CONFIG["toks_per_shard"] // 0.4
for file in dataset_files:
	print(f"{Fore.YELLOW}{Style.BRIGHT}{file}")
	with open(file, "r", encoding="utf-8") as f:
		train_data.extend(json.load(f))

	if round(lensum(train_data) / num_chars_to_enc_at_once, 2) < 0.95:
		continue

	# get total number chars and total number of unique chars
	unique_chars = len(set().union(*set().union(*train_data)))
	num_chars = lensum(train_data)

	# verbose
	total_chars += num_chars
	total_unique_chars += unique_chars
	print(f"{(num_chars/1e6)}M total chars,", f"{unique_chars} unique chars")

	# split train and val data
	if CONFIG["data_division"] < 1:
		val_data = split_data(train_data)
	print(f"{(lensum(train_data)/1e6)}M train chars" + f", {(lensum(val_data)/1e6)}M val chars")

	# encode and post-process train data
	num_train_chars = encode_data(train_data, "train")
	num_train_tokens = lensum(train_data)
	save_data(train_data, "train")
	train_data = []

	# encode and post-process val data
	if CONFIG["data_division"] < 1:
		num_val_chars = encode_data(val_data, "val")
		num_val_tokens = lensum(val_data)
		save_data(val_data, "val")
		del val_data
	print()

	total_train_chars += num_train_chars
	total_train_tokens += num_train_tokens
	total_val_chars += num_val_chars if CONFIG["data_division"] < 1 else num_train_chars
	total_val_tokens += num_val_tokens if CONFIG["data_division"] < 1 else num_train_tokens

print(f"{(total_chars/1e6)}M total chars,", f"{total_unique_chars} total unique chars")
print(f"{(total_train_chars/1e6)}M total train chars" + f", {(total_val_chars/1e6)}M total val chars")
print(f"{(total_train_tokens/1e6)}M total train tokens" + f", {(total_val_tokens/1e6)}M total val tokens")
