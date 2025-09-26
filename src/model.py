from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch, math

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

@dataclass
class Config:
    vocab_size: int = 8192
    block_size: int = 1024
    n_layer: int = 2 # num new layers
    r_layer: int = 2 # num reuse layers
    n_head: int = 4
    n_embd: int = 64
    n_qkv: int = 256
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

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

class Rotary(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.n_embd // config.n_head
        self.base = config.rope_theta
        self.block_size = config.block_size
        self.scaling_factor = config.rope_scaling_factor
        self.ntk_alpha = config.rope_ntk_alpha
        self.ntk_beta = config.rope_ntk_beta

    def _compute_concentration_and_inv_freq(self):
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (torch.arange(0, self.head_dim, 2) / self.head_dim)

        if self.scaling_factor > 1.0:
            concentration = (0.1 * math.log(self.scaling_factor) + 1.0)  # YaRN concentration

            d_half = self.head_dim / 2

            # NTK by parts
            low = (d_half * math.log(self.block_size / (self.ntk_beta * 2 * math.pi)) / math.log(self.base))
            high = (d_half * math.log(self.block_size / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base))
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (torch.arange(d_half) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask

        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def _apply_rotary_emb(x, cos: torch.Tensor, sin: torch.Tensor):
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor):
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = self._apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = self._apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key

class AttentionOnDetail(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_head = config.n_qkv // config.n_head
        self.n_head = config.n_head

        # merged QKV weights
        self.qkv = CastedLinear(config.n_embd, 3*config.n_qkv)
        self.swiglu = CastedLinear(config.n_qkv, 2*config.n_embd)
        self.out = CastedLinear(config.n_embd, config.n_embd)
        self.rotary = Rotary(config)

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
