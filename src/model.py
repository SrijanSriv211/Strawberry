from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch, math

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

@dataclass
class Config:
    vocab_size: int = 8192
    block_size: int = 1024
    r_layer: int = 2 # num reuse layers
    n_layer: int = 2 # num new layers
    n_head: int = 4
    n_embd: int = 64
    n_qkv: int = 256

class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.w = nn.Parameter(torch.empty(out_features, in_features))

        # init params
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.w.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, self.w)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class AttentionOnDetail(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_head = config.n_qkv // config.n_head
        self.n_head = config.n_head

        # merged QKV weights
        self.qkv = CastedLinear(config.n_embd, 3*config.n_qkv)
        self.swiglu = CastedLinear(config.n_qkv, 2*config.n_embd)
        self.out = CastedLinear(config.n_embd, config.n_embd)
        self.rotary = Rotary(self.d_head, config.block_size)

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.qkv(norm(x)).view(B, T, 3*self.n_head, self.d_head).transpose(1, 2).chunk(3, dim=1) # (B, T, nh, hs) -> (B, nh, T, hs)
        q, k = norm(q), norm(k) # QK norm
        q, k = self.rotary(q, k)

        # https://arxiv.org/pdf/2105.14103
        y = torch.sigmoid(q) * torch.cumsum(torch.sigmoid(k) * v, dim=2)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head*self.d_head) # re-assemble all head outputs side by side

        # output projection
        u, v = self.swiglu(y).chunk(2, dim=-1)
        x + self.out(u * F.silu(v))
        return F.relu(x).square()

class Strawberry(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # factorized token embeddings
        self.embed = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd))
        self.blocks = nn.ModuleList([AttentionOnDetail(config) for _ in range(config.n_layer)])
        self.unembed = CastedLinear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        x = norm(self.embed(idx)) # token embeddings of shape (b, t, n_embd)

        for _ in range(self.config.r_layer):
            for block in self.blocks:
                x = block(x)

        logits = self.unembed(norm(x))
        # if we are given some desired targets also calculate the loss
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stream=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # our very first step, pass the initial sequence context to the model
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # pluck the logits at the final step and scale by desired temperature
            # https://github.com/karpathy/nanoGPT/pull/546/
            if temperature == 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            else:
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)
                # live-stream output if True
                if stream is not None:
                    print(stream.decode([idx_next[0].item()]), end="", flush=True)
        if stream is not None:
            print()
        return idx
