from encoder import Encoder
from rich.progress import track
from colorama import Style, Fore, init
import argparse, pickle, json, os

init(autoreset=True)

# python src/prepare_data.py -i data -o data -e bin/cl4k.bin
parser = argparse.ArgumentParser(description="Prepare training & val dataset")
parser.add_argument("-i", help="dataset path", required=True)
parser.add_argument("-o", help="output path", required=True)
parser.add_argument("-e", help="encoder path", required=True)
parser.add_argument("-d", help="train-val data division ratio", type=float, default=0.8)
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

"""
Pretraining dataset
"""
dataset_files = [os.path.join(CONFIG["dataset_path"], i) for i in os.listdir(CONFIG["dataset_path"]) if os.path.isfile(os.path.join(CONFIG["dataset_path"], i))]
dataset_files = sorted(dataset_files, key=os.path.getsize)
lsum = lambda x: sum([len(i) for i in x])

for file in dataset_files:
	with open(file, "r", encoding="utf-8") as f:
		data = json.load(f)

	n_chars = lsum(data)
	for i, x in enumerate(track(data, f"{Fore.WHITE}{Style.BRIGHT}encoding {Fore.WHITE}{Style.DIM}{file}{Style.RESET_ALL}")):
		data[i] = enc.encode(f"{x}<|eot|>\n", allowed_special="all")

	n_toks = lsum(data)
	n_train_toks = int(n_toks * CONFIG["data_division"])
	n_val_toks = n_toks - n_train_toks

	print(f"{(n_chars/1e6)}M total chars,", f"{(n_toks/1e6)}M total tokens")
	print(f"{(n_train_toks/1e6)}M train toks,", f"{(n_val_toks/1e6)}M val tokens")

	filename = os.path.splitext(file)[0] + ".bin"

	with open(filename, "wb") as f:
		pickle.dump({
			"dataset": data,
			"train_toks": n_train_toks,
			"val_toks": n_val_toks
		}, f)
