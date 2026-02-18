from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch, math

@dataclass
class Config:
	vocab_size: int = 8192
	block_size: int = 1024
	n_layer: int = 2 # new layers
	r_layer: int = 2 # reuse layers
	n_head: int = 4
	n_embd: int = 64
	n_qkv: int = 256
	n_ffn: int = 256
	n_experts: int = 4 # total experts
	a_experts: int = 2 # active experts

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
		self.n_embd = config.n_embd
		self.n_qkv = config.n_qkv

		# merged QKV weights & out proj weights
		# fixed/pre-trained attention qkv, swiglu and out weights
		self.w_attn_qkv = CastedLinear(config.n_embd, config.n_qkv*3).weight
		self.w_attn_swiglu = CastedLinear(config.n_qkv, config.n_embd*2).weight
		self.w_attn_out = CastedLinear(config.n_embd, config.n_embd).weight

		# out proj weights
		self.out = CastedLinear(config.n_embd, config.n_embd)

		# tao values
		self.tao = CastedLinear(1, 3).weight

	# normalize here (fixes initial scale)
	def w_norm(self, w, n):
		target_std = n ** -0.5
		return w * (target_std / (w.std() + 1e-8))

	# return the new "old" & "current" weights.
	def forward(self, x, y, w, update=True):
		# retention mechanism produced qkv, swiglu & out proj weights
		w_qkv, w_swiglu, w_out = w

		# swiglu ffn calculation
		u, v = F.linear(y, w_swiglu).chunk(2, dim=-1)
		y = x + F.linear(u * F.silu(v), w_out)

		if not update:
			return y

		# split the tao values into 3, alpha, beta & gamma
		alpha, beta, gamma = torch.abs(self.tao).clamp(1e-8, 1)

		# calculate adjusted-original value
		# compact O & A of shape (B, T, C) -> (B, C, C) -> (C, C) shape
		ao = (u.transpose(1, 2) @ v).mean(dim=0) * (self.n_embd ** -0.5)
		ao = norm(ao)

		# update qkv weights
		n = w_qkv @ ao
		n = F.silu(n)
		w_qkv = w_qkv + self.out(n)
		w_qkv = self.w_norm(w_qkv, self.n_embd) * alpha

		# update swiglu weights
		n = w_swiglu.view(self.n_embd, 2*self.n_qkv).T @ ao
		n = F.silu(n)
		n = self.out(n)
		w_swiglu = w_swiglu + n.T.contiguous().view(2*self.n_embd, self.n_qkv)
		w_swiglu = self.w_norm(w_swiglu, self.n_qkv) * beta

		# update out weights
		n = w_out @ ao
		n = F.silu(n)
		w_out = w_out + self.out(n)
		w_out = self.w_norm(w_out, self.n_qkv) * gamma

		return y, (w_qkv, w_swiglu, w_out)

# calculate AFT attention (https://arxiv.org/pdf/2105.14103)
class AttentionFreeTransformer(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, w_qkv):
		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v = F.linear(norm(x), w_qkv).chunk(3, dim=-1) # (B, T, C)
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
		return x, F.sigmoid(q) * y

# new version of my AttentionOnDetail attention mechanism
class TheExpertAbundance(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.d_head = config.n_qkv // config.n_head
		self.n_head = config.n_head

	def forward(self, x, cos_sin, w_qkv):
		# batch size, sequence length, embedding dimensionality (n_embd)
		B, T, _ = x.size()

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v = F.linear(norm(x), w_qkv).view(B, T, self.n_head, 3*self.d_head).chunk(3, dim=-1) # (B, T, nh, hs)

		# apply rotary embeddings to queries and keys to get relative positional encoding
		cos, sin = cos_sin
		q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
		q, k = norm(q), norm(k) # QK norm

		# make head be batch dim, i.e. (B, T, nh, hs) -> (B, nh, T, hs)
		q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

		# calculate sdpa
		y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

		# re-assemble all head outputs side by side
		return x, y.transpose(1, 2).contiguous().view(B, T, -1)

class Block(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.r_layer = config.r_layer
		self.n_embd = config.n_embd
		self.n_qkv = config.n_qkv

		self.aft = AttentionFreeTransformer()
		self.tea = TheExpertAbundance(config)
		self.retention = RetentionMechanism(config)

	def check(self, i, w, w_prev, w_initial):
		with torch.no_grad():
			diff_norm = (w - w_prev).norm()
			w_norm = w_prev.norm()
			rel_change = (diff_norm / (w_norm + 1e-8)).item()

			cos_sim = F.cosine_similarity(
				w_prev.contiguous().view(1, -1),
				w.contiguous().view(1, -1),
				dim=-1
			).item()

			cos_sim_init = F.cosine_similarity(
				w_initial.contiguous().view(1, -1),
				w.contiguous().view(1, -1),
				dim=-1
			).item()

			# largest singular value
			try:
				s_max = torch.linalg.svdvals(w)[0].item()
			except:
				s_max = float("nan")

			print(
				f"[Retention {i}] "
				f"rel_change={rel_change:.6f} | "
				f"cos(prev)={cos_sim:.6f} | "
				f"cos(init)={cos_sim_init:.6f} | "
				f"s_max={s_max:.4f}"
			)

	def cat(self, w_qkv, w_swiglu, w_out):
		w_swiglu = w_swiglu.reshape(2*self.n_qkv, self.n_embd)
		return torch.cat([w_qkv, w_swiglu, w_out], dim=0)

	def forward(self, x, cos_sin):
		w_qkv, w_swiglu, w_out = self.retention.w_attn_qkv, self.retention.w_attn_swiglu, self.retention.w_attn_out

		# 3 consecutive global-linear attentions,
		# then 1 local-scaled-dot-product attention
		for i in range(self.r_layer):
			x, y = self.tea(x, cos_sin, w_qkv) if (i + 1) % 4 == 0 else self.aft(x, w_qkv)
			x, w = self.retention(x, y, (w_qkv, w_swiglu, w_out))
			w_qkv, w_swiglu, w_out = w

		# one final TEA forward pass
		x, y = self.tea(x, cos_sin, w_qkv)
		return self.retention(x, y, (w_qkv, w_swiglu, w_out), False)

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
