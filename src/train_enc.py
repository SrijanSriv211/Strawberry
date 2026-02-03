from encoder import Encoder
import argparse, os
enc = Encoder()

# python src/train_enc.py -i data/base/data.txt -o bin/cl4k.bin -v 4096
# python src/train_enc.py -i data/base/json -d true -o bin/cl8k.bin -v 8192
parser = argparse.ArgumentParser(description="A powerful text encryption and decryption program.")
parser.add_argument("-i", help="dataset path", required=True)
parser.add_argument("-o", help="output path", required=True)
parser.add_argument("-v", help="vocab size", type=int, required=True)
parser.add_argument("-d", help="is directory", type=bool, default=False)
parser.add_argument("-r", help="text range", type=int, default=100_000_000)
parser.add_argument("-s", help="special tokens", type=list, default=[
	"<|eop|>", # end of prompt
	"<|eor|>", # end of reason
	"<|eot|>", # end of text
	"<|sep|>",
	"<|call|>", # tool call
	"<|sink|>",
	"<|mask|>"
])
args = parser.parse_args()

CONFIG = {
	"dataset_path": args.i,
	"is_dir": args.d,
	"outpath": args.o,
	"vocab_size": args.v - len(args.s), # remove len of special tokens to get vocab size for merging
	"text_range": args.r,
	"special_tokens": args.s
}

dir = os.path.split(CONFIG["outpath"])[0]
if not os.path.isdir(dir):
	os.mkdir(dir)

#* set `vocab_size` in `config.json` 4096
enc.train([CONFIG["dataset_path"], CONFIG["is_dir"]], CONFIG["vocab_size"], text_range=CONFIG["text_range"])
enc.register_special_tokens(*CONFIG["special_tokens"])
enc.save(CONFIG["outpath"])

print("Special Tokens:\n", enc.special_tokens)
