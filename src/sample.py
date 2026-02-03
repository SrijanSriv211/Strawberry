from model import Strawberry, Config
from utils import print_banner
from encoder import Encoder

from colorama import Style, Fore, init
import argparse, torch

def generate(i, e, l=256, t=0.8, f=None, T=None):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# create an instance of Strawberry
	conf = Config(**i["hyperparams"])
	model = Strawberry(conf)

	# remove `_orig_mod.` prefix from state_dict (if it's there)
	state_dict = i["model"]
	unwanted_prefix = '_orig_mod.'

	for k, v in list(state_dict.items()):
		if k.startswith(unwanted_prefix):
			state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

	# load the saved model state_dict
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval() # set the model to evaluation mode

	# compile the model
	torch.compile(model)

	# load the encoder
	enc = Encoder()
	enc.load(e)

	# encode text and generate output
	enctxt = enc.encode(T, allowed_special="all") if T is not None else []
	out = model.generate(
		enctxt, enc.special_tokens["<|sink|>"],
		max_new_tokens=l, temperature=t, top_k=f
	)[0].tolist()
	return enc.decode(out)

if __name__ == "__main__":
	init(autoreset = True)
	print_banner()

	parser = argparse.ArgumentParser(description="A powerful text encryption and decryption program.")
	parse_prompt = parser.add_mutually_exclusive_group()
	parser.add_argument("--model", "-i", help="model path", required=True)
	parser.add_argument("--encoder", "-e", help="encoder path", required=True)
	parser.add_argument("--length", "-l", help="output length", type=int, default=256)
	parser.add_argument("--temperature", "-t", help="output temperature", type=float, default=0.8)
	parser.add_argument("--top_k", "-f", help="output top_k", type=int, default=None)
	parse_prompt.add_argument("--text_prompt", "-T", help="Text input from the command line.")
	args = parser.parse_args()

	# load text from the text file.
	text = args.text_prompt if args.text_prompt else None
	out = generate(torch.load(args.model), args.encoder, args.length, args.temperature, args.top_k, text)
	print(f"{Fore.WHITE}{Style.DIM}```\n{out}\n```\n")
