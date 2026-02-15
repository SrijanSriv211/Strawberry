# Strawberry: Is strawberry a fruit or a vegetable?

<img src="img/strawberry.png" alt="howmanyrsinthewordstrawberry" style="width:100%;">

Strawberry is primarily an early-stage neural network architecture built on top of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project. Currently not much is implemented, however everything inside this repository is enough to train AI models of various sizes.

Strawberry brings several improvements over the standard GPT-2 architecture, such as:
1. Shared stack of layers across recursion steps inspired from Google's Mixture of Recursions [[paper](https://arxiv.org/pdf/2507.10524)]
2. MoE Swiglu FFN & attention mechanism [[paper](https://arxiv.org/pdf/1701.06538)]
3. Apple's Attention Free Transformer [[paper](https://arxiv.org/pdf/2105.14103)]
4. Swiglu based FFN [[paper](https://arxiv.org/pdf/2002.05202)]
5. Modernized architecture: Rotary embeddings and QK-Norm
6. My custom `The Expert Abundance` attention mechanism
7. Shared embedding weights

## The Expert Abundance
MoE-attention mechanism & Swiglu mini-FFN.

- Derive **QKV**, **Swiglu** & **Output projection** weights by the Retention Mechanism's **Update Rule** *(given below)*

```
Q, K, V     -> Token-level Local-Context-MoE Scaled Dot Product Attention		-> Y
Y           -> Concatenate all local mixture of attention parts                 -> Y
Y           -> Swiglu mini-ffn                                                  -> Y
Y           -> X + out(Y)                                                       -> Y
```

> [!NOTE]
> As of now Token-level Local-Context-MoE has not been implemented in The Expert Abundance.

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
		"n_layer": 2, // number of layers you want
		"n_head": 4,
		"n_embd": 64,
		"n_qkv": 256,
		"n_ffn": 256
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
