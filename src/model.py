from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch

@dataclass
class Config:
	vocab_size: int = 8192
	block_size: int = 1024
	r_layer: int = 2 # reuse layers
	n_layer: int = 2 # new layers
	n_head: int = 4
	n_embd: int = 64
	n_qkv: int = 256

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

class RetentionMechanism(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		assert config.n_qkv % config.n_embd == 0
		self.f_qkv = config.n_qkv // config.n_embd
		self.n_embd = config.n_embd
		self.n_qkv = config.n_qkv

		self.swiglu = CastedLinear(config.n_embd, config.n_embd*2)
		self.out = CastedLinear(config.n_embd, config.n_embd)

	def forward(self, x):
		# swiglu ffn calculation
		u, v = self.swiglu(norm(x)).chunk(2, dim=-1)
		y = x + self.out(u * F.silu(v))

		# update weights
		uv = (u.transpose(1, 2) @ F.silu(v)).mean(dim=0)
		m = torch.arange(1, (self.f_qkv * 6)/2).unsqueeze(1).unsqueeze(1)

		# rms norm & [-pi, pi] norm
		uv_norm = norm(uv).unsqueeze(0)
		uv_ones = torch.ones(uv_norm.shape)
		uv_pi = F.tanh(uv) * 3.14 * m

		# [uv, sin(uv), sin(2*uv), sin(3*uv), ..., cos(uv), cos(2*uv), cos(3*uv), ...]
		uv = torch.cat([uv_ones, uv_norm, torch.sin(uv_pi), torch.cos(uv_pi)], dim=0)

		# swiglu ffn calculation
		u, v = self.swiglu(uv).chunk(2, dim=-1)
		uv = uv + self.out(u * F.silu(v))
		uv = uv.view(6 * self.n_qkv, self.n_embd)
		uv = uv * (uv.size(0) ** -0.5)

		# retention mechanism produced qkv, attn out proj & swiglu weights
		w_qkv, w_attn_out, w_swiglu = torch.split(uv, [3*self.n_qkv, self.n_qkv, 2*self.n_qkv], dim=0)
		return y, (w_qkv, w_attn_out.T, w_swiglu)

# new version of my AttentionOnDetail attention mechanism
class TheExpertAbundance(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.d_head = config.n_qkv // config.n_head
		self.n_head = config.n_head

	# calculate AFT attention (https://arxiv.org/pdf/2105.14103)
	def aft(self, qkv):
		q, k, v = qkv.chunk(3, dim=-1) # (B, T, C)
		q, k = norm(q), norm(k) # QK norm

		# exponentiate keys
		w = torch.exp(k)
		kv = w * v

		# causal cumulative sums
		w = torch.cumsum(w, dim=1)
		kv = torch.cumsum(kv, dim=1)

		# normalize
		y = kv / (w + 1e-6)

		# gate with query
		return F.sigmoid(q) * y

	def spda(self, B, T, qkv, cos_sin):
		q, k, v = qkv.view(B, T, self.n_head, 3*self.d_head).chunk(3, dim=-1) # (B, T, nh, hs)

		# apply rotary embeddings to queries and keys to get relative positional encoding
		cos, sin = cos_sin
		q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
		q, k = norm(q), norm(k) # QK norm

		# make head be batch dim, i.e. (B, T, nh, hs) -> (B, nh, T, hs)
		q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

		# calculate sdpa
		y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

		# re-assemble all head outputs side by side
		return y.transpose(1, 2).contiguous().view(B, T, -1)

	def forward(self, x, cos_sin, w_qkv, w_out, i):
		# batch size, sequence length, embedding dimensionality (n_embd)
		B, T, _ = x.size()

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		qkv = F.linear(norm(x), w_qkv)
		attn = self.spda(B, T, qkv, cos_sin) if (i+1) % 4 == 0 else self.aft(qkv)
		out = F.linear(attn, w_out)
		return x + out

class Swiglu(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.out = CastedLinear(config.n_qkv, config.n_embd)

	def forward(self, x, w_swiglu):
		u, v = F.linear(norm(x), w_swiglu).chunk(2, dim=-1)
		return x + self.out(u * F.silu(v))

class Block(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.r_layer = config.r_layer

		self.tea = TheExpertAbundance(config)
		self.retention = RetentionMechanism(config)
		self.swiglu = Swiglu(config)

	def forward(self, x, cos_sin):
		for i in range(self.r_layer):
			x, (w_qkv, w_attn_out, w_swiglu) = self.retention(x)
			x = self.tea(x, cos_sin, w_qkv, w_attn_out, i)
			x = self.swiglu(x, w_swiglu)
		return x

class Strawberry(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		assert config.vocab_size is not None
		assert config.block_size is not None
		self.config = config

		# factorized token embeddings
		self.embed = nn.Embedding(config.vocab_size, config.n_embd)
		self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
		self.unembed = CastedLinear(config.n_embd, config.vocab_size)
		self.embed.weight = self.unembed.weight

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

		# token embeddings of shape (b, t, n_embd)
		x = self.embed(idx)
		x = norm(x)

		for block in self.blocks:
			x = block(x, cos_sin)

		# forward the lm_head (compute logits)
		x = norm(x)
		logits = self.unembed(x)

		# if we are given some desired targets also calculate the loss
		loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction="mean")
		return logits, loss

	@torch.no_grad()
	def generate(self, idx, sink_tok, max_new_tokens, temperature=1.0, top_k=None):
		sink_tok = torch.tensor([sink_tok], dtype=torch.int64).unsqueeze(0)
		idx = torch.tensor(idx, dtype=torch.int64).unsqueeze(0)

		for _ in range(max_new_tokens):
			# our very first step, pass the initial sequence context to the model
			# if the sequence context is growing too long we must crop it at block_size
			idx_cond = idx[:, -self.rotary_block_size:] if idx.size(1) > self.rotary_block_size else idx
			idx_cond = torch.cat([sink_tok, idx_cond], dim=1)

			# forward the model to get the logits for the index in the sequence
			logits, _ = self(idx_cond)
			logits = logits[:, -1, :]

			# https://github.com/karpathy/nanoGPT/pull/546/
			# pluck the logits at the final step and scale by desired temperature
			if temperature > 0:
				logits = logits / temperature

				# optionally crop the logits to only the top k options
				if top_k is not None:
					v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
					logits[logits < v[:, [-1]]] = -float("Inf")

				# apply softmax to convert logits to (normalized) probabilities,
				# sample from the distribution and,
				probs = F.softmax(logits, dim=-1)
				idx_next = torch.multinomial(probs, num_samples=1)

			else:
				idx_next = torch.argmax(logits, dim=-1, keepdim=True)

			# replace MASK with predicted token
			idx = torch.cat([idx, idx_next], dim=1)
		return idx
