# Strawberry: Is strawberry a fruit or a vegetable?

<img src="img/strawberry.png" alt="howmanyrsinthewordstrawberry" style="width:100%;">

Strawberry is primarily an early-stage neural network architecture built on top of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project. Currently not much is implemented, however everything inside this repository is enough to train AI models of various sizes.

Strawberry brings several improvements over the standard GPT-2 architecture, such as:
1. Shared stack of layers across recursion steps inspired from Google's Mixture of Recursions [[paper](https://arxiv.org/pdf/2507.10524)]
2. `N:1` ratio attention placement inspired from Kimi Linear [[paper](https://arxiv.org/pdf/2510.26692)]
3. MoE Swiglu FFN & attention mechanism [[paper](https://arxiv.org/pdf/1701.06538)]
4. Apple's Attention Free Transformer [[paper](https://arxiv.org/pdf/2105.14103)]
5. Swiglu based FFN [[paper](https://arxiv.org/pdf/2002.05202)]
6. Modernized architecture: Rotary embeddings and QK-Norm
7. My custom `The Expert Abundance` attention mechanism
8. My custom `Retention Mechanism` architecture
9. Shared embedding weights

## The Expert Abundance
MoE-attention mechanism & Swiglu mini-FFN.

- Derive **QKV**, **Swiglu** & **Output projection** weights by the Retention Mechanism's **Update Rule** *(given below)*

```
X, Q, K, V  -> Token-level Local-Context-MoE Scaled Dot Product Attention		-> Y
Y           -> Concatenate all local mixture of attention parts                 -> X, Y
```

> [!NOTE]
> As of now Token-level Local-Context-MoE has not been implemented in The Expert Abundance.

## Retention Mechanism
1. Derive **QKV**, **Swiglu** & **out projection** weights using the given input.
2. Replacing the standard FFN layer, the **Retention** and is placed after the **Attention** layer.
3. The retention layer first performs the same computation as a regular Swiglu FFN, then continues to generate new **QKVO** weights using those computations.

### Base Weights
Each block contains 5 learned tensors.
- `w_attn_qkv` 		-> shape `(C, 3D)`
- `w_attn_swiglu` 	-> shape `(2C, D)`
- `w_attn_out` 		-> shape `(C, C)`
- `out` 			-> shape `(C, C)`
- `tao` 			-> shape `(1, 3)`

where
- `C` = embedding dimension (`n_embd`)
- `D` = QKV dimension (`n_qkv`)

`w_attn_...` act as initial attention, swiglu & swiglu_out projection weights

We initialize these weights in following way:
- `w_qkv = w_attn_qkv`
- `w_swiglu = w_attn_swiglu`
- `w_out = w_attn_out`

`w_qkv`, `w_swiglu` and `w_out` are updated iteratively `r_layer` number of times.

### Architecture Design
For each retention step:
```
x, y = Attention(x, w_qkv)
x, (w_qkv, w_swiglu, w_out) = Retention(x, y, (w_qkv, w_swiglu, w_out))
```

- `Attention` alternates between `AttentionFreeTransformer` and `TheExpertAbundance`.
- **Attention Free Transformer** (global-context linear attention) is applied for 3 steps.
- **The Expert Abundance** (local-context-sliding-window scaled-dot-product attention) is then applied in every 4th step.
- Giving `3:1` AFT-to-TEA ratio. This design is inspired by **Kimi Linear's** `N:1` KDA-to-MLA ratio.
- After `r_layer` updates, one final pass applies the **The Expert Abundance** attention mechanism & **Retention Mechanism** without further weight modification.

### Swiglu FFN
The **Swiglu FFN** is built into the Retention Mechanism as it's outputs are used as a basis for weights updates.

Given attention input & output `x` & `y` respectively:
```
u, v = y @ w_swiglu.T
y = x + (u * silu(v)) @ w_out.T
```

### Calculate orignal-adjust value
Use `u` & `v` to create a compact `(C, C)` shaped input-dependent tensor
- `u` act similar to Query weights in attention mechanism.
- `v` tell the model how to adjust the information presented by `u`.
- It is then used to calculate the **orignal-adjust** value.

It happens in the following way:
- `u` & `v` share the same shape `(B, T, C)`.
- `u.transpose(1, 2) @ v` produces the **oa** tensor of shape `(B, C, C)`.
- `mean(dim=0)` averages across batch, and produces a tensor of shape `C, C`.
- `RMSnorm` is applied to normalize the newly generated **oa** tensor.

### Update Rule
`w_qkv` and `w_out` are updated in the following way:
```
n = w @ oa
n = F.silu(n)
w = w + out(n)
w = norm(w, C) * (alpha or gamma)
```

`w_swiglu` is updated in the following way:
```
n = w.view(C, 2*D).T @ oa
n = F.silu(n)
n = out(n)
w = w + n.T.contiguous().view(2*C, D)
w = norm(w, D) * beta
```

- `tao` is split into 3 learned positive-only scalars, `alpha`, `beta` & `gamma`.
- `alpha`, `beta` & `gamma` are used in updating `w_qkv`, `w_swiglu` & `w_out` respectively.
- After this `y` from Swiglu FFN & `(w_qkv, w_swiglu, w_out)` are returned

## Getting Started
<ins>**1. Downloading the repository:**</ins>

Start by cloning the repository with `git clone https://github.com/SrijanSriv211/Strawberry`.

<ins>**2. Configuring the hyperparameters:**</ins>

The configuration object can be found in `train.py`, which can also be copy-pasted into a separate `.json` file

```json
{
	"dataset": {
		"data_division": 0.8, // 80% train data, 20% val data
		"load_from_file": true, // if false it'll try to load all files in the given path (assuming the given path in that case is a directory)
		"path": "data/webtext.bin"
	},
	"checkpoints": {
		"path": "bin/ck",
		"interval": 2000,
		"create_checkpoints": true
	},
	"model_hyperparams": {
		"vocab_size": 8192,
		"block_size": 256,
		"r_layer": 2, // number of retention update steps
		"n_layer": 2, // number of layers
		"n_head": 4,
		"n_embd": 64,
		"n_qkv": 256
	},
	"optimizer_hyperparams": {
		"eps": 1e-10,
		"beta1": 0.9,
		"beta2": 0.95,
		"weight_decay": 1e-1,
		"use_muon": false, // for the muon optimizer, it's code is present in `optimizer.py`
		"momentum": 0.95 // muon optimizer's hyperparameter
	},
	"model_path": "bin/air.strawberry", // model's save path after training
	"encoder_path": "bin/cl8k.bin", // load encoder from the given path
	"init_from": "scratch", // "scratch" -> init a fresh model and train; "<model_path>,pretrained" -> load pre-trained model from path `<model_path>` and train
	"seed": "auto",

	"gradient_accumulation_steps": 1,
	"batch_size": 4,

	"max_iters": 50000,
	"eval_interval": 2000,
	"log_interval": 200,
	"eval_iters": 200,

	"decay_lr": true,
	"lr_decay_iters": 50000,
	"learning_rate": 3e-3,
	"cooldown_frac": 0.4,
	"warmup_iters": 2000,
	"min_lr": 3e-4
}
```

## Citation

```
@software{Strawberry,
    author={Srijan Srivastava},
    title={Strawberry},
    url={https://github.com/SrijanSriv211/Strawberry},
    version={0.1.0},
    year = {2026}
}
```

<img src="img/rdr2.png" alt="lookwhosback" style="width:100%;">
