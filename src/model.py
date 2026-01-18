from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch

@dataclass
class Config:
    vocab_size: int = 8192
    block_size: int = 1024
    r_layer: int = 2 # num reuse layers
    n_layer: int = 2 # num new layers
    n_head: int = 4
    n_embd: int = 64
    n_qkv: int = 256
    n_experts: int = 4
    experts_per_tok: int = 2

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # init params
        self.reset_parameters()

    # transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
    def reset_parameters(self):
        s = 3**0.5 * self.in_features**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        with torch.no_grad():
            self.weight.uniform_(-s, s)

    def forward(self, x):
        return F.linear(x, self.weight)

# new version of the old AttentionOnDetail
class TheExpertAbundance(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_head = config.n_qkv // config.n_head
        self.n_head = config.n_head

        # experts
        self.experts_per_tok = config.experts_per_tok
        self.n_experts = config.n_experts
        self.gate = CastedLinear(config.n_embd, self.n_experts)

        # merged QKV weights
        self.qkv_1 = CastedLinear(config.n_embd, 3*config.n_qkv)
        self.qkv_2 = CastedLinear(self.d_head, 3*self.d_head)

        # out projection weights
        self.swiglu = CastedLinear(config.n_qkv, 2*config.n_embd)
        self.out = CastedLinear(config.n_embd, config.n_embd)

        # zero out projection weights in all blocks
        torch.nn.init.zeros_(self.out.weight)

    def forward(self, x, cos_sin):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, _ = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.qkv_1(norm(x)).view(B, T, 3*self.n_head, self.d_head).chunk(3, dim=2) # (B, T, nh, hs)

        # apply rotary embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm

        # make head be batch dim, i.e. (B, T, nh, hs) -> (B, nh, T, hs)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # # calculate AFT attention
        # y = torch.softmax(q) * torch.cumsum(torch.softmax(k) * v, dim=2) # https://arxiv.org/pdf/2105.14103

        # calculate sdpa
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, -1) # re-assemble all head outputs side by side

        # output projection
        u, v = self.swiglu(y).chunk(2, dim=-1)
        return x + self.out(u * F.silu(v))

class Strawberry(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # factorized token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([TheExpertAbundance(config) for _ in range(config.n_layer)])

        # to support meta device initialization, we init the rotary embeddings here, but it's fake
        # as for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # in the future we can dynamically grow the cache, for now it's fine.
        self.rotary_block_size = config.block_size * 10 # 10X over-compute should be enough, TODO make nicer?
        d_head = config.n_qkv // config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_block_size, d_head)

    def _precompute_rotary_embeddings(self, block_size, d_head, base=10000):
        # stride the channels
        channel_range = torch.arange(0, d_head, 2)
        inv_freq = 1.0 / (base ** (channel_range / d_head))
        # stride the time steps
        t = torch.arange(block_size)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting

    def forward(self, idx, targets=None):
        B, T = idx.size()

        # grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, :+T], self.sin[:, :+T]

        x = self.embed(idx) # token embeddings of shape (b, t, n_embd)
        x = norm(x)

        for block in self.blocks:
            for _ in range(self.config.r_layer):
                x = block(x, cos_sin)

        x = norm(x)
        logits = F.linear(x, self.embed) # tying embed & unembed weights by using embed weights to unembed `x`

        # forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        # if we are given some desired targets also calculate the loss
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction="mean")
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
