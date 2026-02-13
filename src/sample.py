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
	parser.add_argument("--model", "-i", help="model path", required=True)
	parser.add_argument("--encoder", "-e", help="encoder path", required=True)
	parser.add_argument("--length", "-l", help="output length", type=int, default=256)
	parser.add_argument("--temperature", "-t", help="output temperature", type=float, default=0.8)
	parser.add_argument("--top_k", "-f", help="output top_k", type=int, default=None)
	parser.add_argument("--text_prompt", "-T", help="Text input from the command line.", default=None)
	args = parser.parse_args()

	texts = [
		"Steve Jobs made the soul of Apple",
		"Red Dead Redemption 2 is the best game of all time",
		"Arthur Morgan: We are bad men, but we ain't them",
		"ENZO: I was willing to die for this family. But now I have to live for my own.\nISABELLA:",
		"LLMs like GPT-3 and GPT 4 show remarkable scaling laws",
		"GTA 3 is one of the loneliest Grand Theft Auto game ever made",
		"This language model is just plain dumb",
		"Google ",
		"Hello I'm a language model, and ",
		"Can I say that Calcia is really a branch of math or is it something nonsense",
		"Every year the moon is going",
		"o/ The workings of the Undetailed",
		"import flask",
		"int main(int argc, char const *argv[])\n{",
		"<body>\n",
		"Compose a 1-3 sentence description of a run-down apartment.<|eop|>",
		"Provide a scalar estimation on the following statement: Movies adapted from books are often better than the",
		"Create a program that generate a 100 to 500-word long text summarizing the content of the given",
		"How does the endocannabinoid system affect the immune system?",
		"You are given two sentences, combine them to create a new sentence: I'm curious.",
		"Describe three steps involved in the process of photosynthesis",
		"Describe the process of decision tree learning.<|eop|>\nDecision tree learning is a supervised machine learning",
		"Classify this statement as optimistic or pessimistic: Life is filled with disappointments.<|eop|>",
		"What is 2+2<|eop|>",
		"Tell me something about spacetime<|eop|>",
		"Why do we think that we can't think if we couldn't think<|eop|>",
		"",
		"",
		""
	] if args.text_prompt is None else [args.text_prompt]

	for text in texts:
		print(f"{Fore.WHITE}{Style.BRIGHT}> {text}")

		if text == "/q":
			break

		elif text.strip() == "":
			text = None

		out = generate(torch.load(args.model), args.encoder, args.length, args.temperature, args.top_k, text)
		print(f"{Fore.WHITE}{Style.DIM}{out}\n")
