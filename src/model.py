from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn, torch

@dataclass
class Config:
	vocab_size: int = 8192
	block_size: int = 1024
	n_layer: int = 2 # new layers
	n_head: int = 4
	n_embd: int = 64

def norm(x):
	return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
	assert x.ndim == 4  # multihead attention
	d = x.shape[3] // 2
	x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
	y1 = x1 * cos + x2 * sin # rotate pairs of dims
	y2 = x1 * (-sin) + x2 * cos
	return torch.cat([y1, y2], 3)

# """
# "Pro" in `ProLinear` stands for Progression, from Arithmetic progression.

# I came up with this idea when I was analyzing weights of already pretrained models.
# I found out that when you sort all the parameters of a linear layer you get a pretty smooth fall. Something like this:

# ```
# [1.2345, 0.987, 0.9723, 0.8723, 0.5149, ..., -0.963]
# ```

# When you take it's mean, the mean's between 0.01-0.5.
# If you take 2 values and subtract the prior to the latter in such a way `1.2345 - 0.987` you get values like this 0.2475 (for example in this case).
# These values can range anywhere between [-1, 1] including both ends.
# Basically the parameters of a model can compressed into a much much much smaller list of numbers and can be constructed back.

# `ProLinear` constructs the weight on this exact same idea. Have a starting value, randomly select a difference,
# add that difference to the starting value, append it to a list, take the last value of the list,
# randomly select a difference and continue from here.

# After that shuffle the list then reshape the list into the desired shape.

# Here all the values, starting value, differences, and seeds for both random selector of difference and shuffler are trainable,
# so that the model can adjust them on their own.
# """
# class ProLinear(nn.Module):
# 	def __init__(self, in_features, out_features, size=512):
# 		super().__init__()
# 		self.in_features = in_features
# 		self.out_features = out_features
# 		self.size = size

# 		self.diffs = nn.Parameter(torch.empty(self.size))
# 		self.gen = torch.Generator()

# 		self.reset_parameters()

# 	# transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
# 	def reset_parameters(self):
# 		n = self.in_features * self.out_features
# 		s = 3**0.5 * self.in_features**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
# 		with torch.no_grad():
# 			self.diffs.uniform_(-s / n**0.5, s / n**0.5) # cumsum over n steps amplifies by sqrt(n), so we shrink steps to compensate

# 	def construct_weight(self):
# 		self.gen.manual_seed(42)

#         # sample which difference to add at each step
# 		n = self.in_features * self.out_features
# 		indices = torch.randint(0, self.diffs.size(0), (n - 1,), generator=self.gen)

#         # build the AP-style list: start, start+d0, start+d0+d1, ...
# 		selected_diffs = self.diffs[indices]
# 		cumulative = torch.cumsum(selected_diffs, dim=0)
# 		weight = cumulative[torch.randperm(n, generator=self.gen)]
# 		weight = weight * 3**0.5 * self.in_features**-0.5

# 		return weight.view(self.out_features, self.in_features)

# 	def forward(self, x):
# 		return F.linear(x, self.construct_weight())

class AttentionOnDetail(nn.Module):
	def __init__(self, config: Config, chunk=1):
		super().__init__()
		self.n_head = config.n_head
		n_qkv = config.n_embd * self.n_head
		self.T0 = 16 # number of tokens in each chunk

		self.sink = nn.Parameter(torch.empty(1, self.n_head, config.n_embd))
		self.qkvg = nn.Linear(config.n_embd, 4*n_qkv, bias=False)
		self.patch = nn.Linear(self.T0 * config.n_embd, 1, bias=False).weight

		self.out = nn.Linear(n_qkv, config.n_embd*chunk, bias=False)
		self.tao = nn.Parameter(torch.tensor([1.2, 1.2]))

	def patch_toks(self, x):
		# batch size, sequence length, embedding dimensionality (n_embd)
		B, T, C = x.size()

		# context already smaller or of-the-size of the patch, so don't patch and just move on.
		if T <= self.T0:
			return x, None, None, 0

		# calculate num pad tokens
		P = (self.T0 - (T % self.T0)) % self.T0

		# calculate qkv & pad `x` if the it's not properly divisible
		x0 = torch.cat([torch.zeros(B, P, C), x], dim=1) if P > 0 else x
		x0 = x0.view(B, (T + P) // self.T0, -1) # (B, T+P, C) -> (B, (T + P) // T0, T0*C)

		# (B, (T + P) // T0)
		y = norm(x0) @ self.patch.view(-1)
		y = torch.softmax(y, dim=-1)
		i = y.topk(self.T0 // 4, dim=-1).indices # select the top `T0/4` indices
		i = i.sort(dim=-1).values # sort indices from first -> last

		# bottom `K` indices sorted from first to last
		i0 = y.topk(self.T0 * 3 // 4, dim=-1, largest=False).indices
		i0 = i0.sort(dim=-1).values

		# return input `x` and topk indices/patches
		return x0.view(B, -1, C), i, i0, P

	def forward(self, x, cos_sin):
		# batch size, sequence length, embedding dimensionality (n_embd)
		B, T, C = x.size()

		# group tokens into multiple patches and select topk most relevant patchs
		x0, I, I0, P = self.patch_toks(x)

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		qkvg = self.qkvg(x0)

		# (B, K, nh, hs)
		if I is not None:
			q, k, v, g = qkvg.view(B, (T + P) // self.T0, -1).chunk(4, dim=-1) # (B, K, nh*hs)
			batch_idx = torch.arange(B).unsqueeze(1)

			# (B, K, nh, hs)
			q0 = q[batch_idx, I0, ...].view(B, -1, self.n_head, C)
			q = q[batch_idx, I, ...].view(B, -1, self.n_head, C)
			k = k[batch_idx, I, ...].view(B, -1, self.n_head, C)
			v = v[batch_idx, I, ...].view(B, -1, self.n_head, C)
			g = g.contiguous().view(B, -1, self.n_head, C)

		else:
			q, k, v, g = qkvg.view(B, T, self.n_head -1).chunk(4, dim=-1) # (B, T, nh, hs)
			q0 = None

		# apply attention sink to the sequence
		s = self.sink.repeat_interleave(B, dim=0).unsqueeze(1)
		q = torch.cat([s, q], dim=1)
		k = torch.cat([s, k], dim=1)
		v = torch.cat([s, v], dim=1)

		print(x.shape, x0.shape, q.shape, q0.shape, g.shape)
		import sys; sys.exit()

		# apply rotary embeddings to queries and keys to get relative positional encoding
		cos, sin = cos_sin
		q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
		q, k = norm(q), norm(k) # QK norm

		# sharper attention
		q = q * self.tao[0]
		k = k * self.tao[1]

		# make head be batch dim, i.e. (B, T, nh, hs) -> (B, nh, T, hs)
		q, k, v, g = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), g.transpose(1, 2)

		# calculate sdpa
		y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)[:, :, 1:, :]

		# apply gated attention
		# https://arxiv.org/pdf/2505.06708
		y = y * F.sigmoid(g)

		# re-assemble all head outputs side by side and remove attention sink from the sequence
		y = y.transpose(1, 2).contiguous().view(B, T, -1)
		return self.out(y)

class Silia(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.attn1 = AttentionOnDetail(config, 2)
		self.attn2 = AttentionOnDetail(config)

	def forward(self, x, cos_sin):
		u, v = self.attn1(norm(x), cos_sin).chunk(2, dim=-1)
		y = u * F.silu(v)
		return self.attn2(y, cos_sin)

class AttnResidual(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.q = nn.Parameter(torch.zeros(config.n_embd)) # (C)

	def forward(self, x, s):
		# (N, B, T, C)
		stack = torch.cat([s, x.unsqueeze(0)], dim=0)
		k = norm(stack)

		# (N, B, T)
		y = k @ self.q
		y = torch.softmax(y, dim=0)
		y = torch.einsum("NBT,NBTC->BTC", y, stack) # (B, T, C)
		return y, stack

class Block(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		self.silia = Silia(config)
		self.residual = AttnResidual(config)

	def forward(self, x, cos_sin, stack):
		y = self.silia(x, cos_sin)
		return self.residual(y, stack)

class Strawberry(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		assert config.vocab_size is not None
		assert config.block_size is not None
		self.config = config

		# factorized token embeddings
		self.embed = nn.Embedding(config.vocab_size, config.n_embd)
		self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
		self.unembed = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		self.embed.weight = self.unembed.weight

		# to support meta device initialization, we init the rotary embeddings here, but it's fake
		# as for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
		# so let's just over-compute them, but assert fail if we ever reach that amount.
		# in the future we can dynamically grow the cache, for now it's fine.
		self.rotary_block_size = config.block_size * 10 # 10X over-compute should be enough, TODO make nicer?
		cos, sin = self._precompute_rotary_embeddings(self.rotary_block_size, config.n_embd)
		self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
		self.register_buffer("sin", sin, persistent=False)

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
		cos_sin = self.cos[:, :+T+1], self.sin[:, :+T+1]

		# token embeddings of shape (b, t, n_embd)
		x = self.embed(idx)
		x = norm(x)

		stack = x.unsqueeze(0)
		for block in self.blocks:
			x, stack = block(x, cos_sin, stack)

		# forward the lm_head (compute logits)
		x = norm(x)
		logits = self.unembed(x)

		# if we are given some desired targets also calculate the loss
		loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction="mean")
		return logits, loss

	@torch.no_grad()
	def generate(self, idx, max_new_tokens, device, temperature=1.0, top_k=None):
		idx = torch.tensor(idx, dtype=torch.int64, device=device).unsqueeze(0)

		for _ in range(max_new_tokens):
			# our very first step, pass the initial sequence context to the model
			# if the sequence context is growing too long we must crop it at block_size
			idx_cond = idx[:, -self.rotary_block_size:] if idx.size(1) > self.rotary_block_size else idx

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
			idx = torch.cat([idx, idx_next], dim=1)
		return idx
