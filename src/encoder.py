from colorama import init, Fore, Style
from collections import Counter
import pickle, regex, json, time, os

init(autoreset=True)

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT4_SPLIT_PATTERN = "|".join(
	[
		r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
		r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
		r"""\p{N}{1,3}""",
		r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
		r"""\s*[\r\n]+""",
		r"""\s+(?!\S)""",
		r"""\s+""",
	]
)

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

def get_text(name, is_dir):
	files = [os.path.join(name, i) for i in os.listdir(name)] if is_dir else [name]
	text = []
	for file in files:
		with open(file, "r", encoding="utf-8") as f:
			text.extend([f.read()] if file.endswith(".txt") else json.load(f))
	return "\n".join(text)

class Encoder:
	def __init__(self, pattern=None):
		"""
		- pattern: optional string to override the default (GPT-4 split pattern)
		- special_tokens: str -> int dictionary of special tokens
			example: {'<|endoftext|>': 100257}
		"""
		self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
		self.compiled_pattern = regex.compile(self.pattern)
		self.special_tokens = {}
		self.inverse_special_tokens = {}
		self.vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

	def train(self, path, vocab_size=256, text_range=None):
		"""
		- path: [name, is_dir]
		- vocab_size: max number of merges to be made - 256 bytes
		- text_range: how much many chars from the text should be used to train (default: None means entire text)
		"""
		assert vocab_size >= 256
		start_time = time.time()

		text = get_text(*path)
		print(
			"encoding text with", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6}M", "total characters and",
			f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters"
		)

		if text_range is not None:
			text = text[:text_range]
			print(
				"ranged text has", f"{Fore.WHITE}{Style.BRIGHT}{len(text)/1e6}M", "characters and",
				f"{Fore.WHITE}{Style.BRIGHT}{len(set(text))}", "unique characters"
			)

		# pre-processing text
		print(f"encoding text chunks... {Fore.WHITE}{Style.DIM}(takes a ~minute)")

		# split the text up into text chunks
		chunks = regex.findall(self.compiled_pattern, text)
		del text

		chk_count = {tuple(i.encode("utf-8")): x for i, x in Counter(chunks).most_common() if len(i) > 1}
		ids = [list(i.encode("utf-8")) for i in sorted(set(chunks), key=len) if len(i) > 1]

		print(dict(Counter([
			pair
			for i in [list(i.encode("utf-8")) for i in chunks if len(i) > 1]
			for pair in zip(i, i[1:])
		]).most_common()), "\n")

		pairs = [pair for i in ids for pair in zip(i, i[1:])]
		a = Counter(pairs).most_common()
		b = [str(i)[1:-1] for i in chk_count.keys()]
		stats = {}
		for i, x in a:
			k = str(i)[1:-1]
			c = [(chk_count[tuple([int(m) for m in j.split(", ")])], j.count(k)) for j in b if k in j]
			n = x - len(c) + sum([1 for _, e in c if e > 1]) + sum([d for d, _ in c])
			stats[i] = n
		stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
		pair = sorted(stats.items(), key=lambda x: x[1], reverse=True)[0]

		print(stats)
		print()
		print(chk_count)
		print()
		print(pair)

enc = Encoder()
# enc.train(["data/json/webtext.json", False], 8192)
enc.train(["data/json/test.txt", False], 8192)
