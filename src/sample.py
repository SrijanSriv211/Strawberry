from colorama import Style, Fore, init
from encoder import Encoder
from model import Config, Strawberry
import warnings, argparse, torch, sys, os

# supress pytorch's future warning:
# You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly.
# It is possible to construct malicious pickle data which will execute arbitrary code during unpickling
# (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details).
# In a future release, the default value for `weights_only` will be flipped to `True`.
# This limits the functions that could be executed during unpickling.
# Arbitrary objects will no longer be allowed to be loaded via this mode
# unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`.
# We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file.
# Please open an issue on GitHub for any issues related to this experimental feature.
warnings.filterwarnings("ignore", category=FutureWarning)
init(autoreset = True)

# check for the input source and append all the input texts to the inputs list.
def load_text(filepath):
    if os.path.isfile(filepath) == False:
        print(f"{filepath}: no such file or directory")
        sys.exit()

    with open(filepath, "r", encoding="utf-8") as f:
        return [i.strip() for i in f.readlines()]
    return []

def prepare_context(encoded_text, device):
    if encoded_text == None:
        return torch.zeros((1, 1), dtype=torch.long, device=device)

    return torch.tensor(encoded_text, dtype=torch.long, device=device).unsqueeze(0)

def generate(i, e, l=256, t=0.8, f=None, s=False, d="auto", T=[None]):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d

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

    enc = Encoder()
    enc.load(e)

    for txt in T:
        enctxt = enc.encode(txt, allowed_special="all") if txt != None else txt
        out = model.generate(prepare_context(enctxt, device), max_new_tokens=l, temperature=t, top_k=f, stream=enc if s else None)[0].tolist()
    return enc.decode(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A powerful text encryption and decryption program.")
    parse_prompt = parser.add_mutually_exclusive_group()
    parser.add_argument("--model", "-i", help="model path", required=True)
    parser.add_argument("--encoder", "-e", help="encoder path", required=True)
    parser.add_argument("--length", "-l", help="output length", type=int, default=256)
    parser.add_argument("--temperature", "-t", help="output temperature", type=float, default=0.8)
    parser.add_argument("--top_k", "-f", help="output top_k", type=int, default=None)
    parser.add_argument("--stream", "-s", help="stream output", type=bool, default=False)
    parser.add_argument("--device", "-d", help="device", type=str, default="auto")
    parse_prompt.add_argument("--text_prompt", "-T", help="Text input from the command line.")
    parse_prompt.add_argument("--file_prompt", "-F", help="Takes a text file as an input.")
    args = parser.parse_args()

    # load text from the text file.
    text = [args.text_prompt] if args.text_prompt else load_text(args.file_prompt) if args.file_prompt else [None]
    out = generate(torch.load(args.model), args.encoder, args.length, args.temperature, args.top_k, args.stream, args.device, text)
    print(f"{Fore.WHITE}{Style.DIM}```\n{out}\n```\n") if not args.stream else None
