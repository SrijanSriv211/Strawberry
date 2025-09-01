from colorama import init, Fore, Style
import pickle, regex, json, time, os

init(autoreset=True)

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def get_text(name, is_dir):
	files = [os.path.join(name, i) for i in os.listdir(name)] if is_dir else [name]
	text = []
	for file in files:
		with open(file, "r", encoding="utf-8") as f:
			text.extend([f.read()] if file.endswith(".txt") else json.load(f))
	return "\n".join(text)
