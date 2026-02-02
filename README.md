# Strawberry: Is strawberry a fruit or a vegetable?
- Strawberry is a model architecture.
- It brings several improvements over the standard Transformer architecture, such as:
    1. Shared stack of layers across recursion steps inspired from Google's Mixture of Recursions [[paper](https://arxiv.org/pdf/2507.10524)]
    2. Modernized architecture: Rotary embeddings and QK-Norm
    3. Shared embedding weights
    4. Apple's Attention Free Transformer [[paper](https://arxiv.org/pdf/2105.14103)]
    5. Swiglu based FFN [[paper](https://arxiv.org/pdf/2002.05202)]
    5. MoE Swiglu FFN & attention mechanism [[paper](https://arxiv.org/pdf/1701.06538)]
    6. My custom `Retention Mechanism` architecture
    7. My custom `The Expert Abundance` attention mechanism

## Retention Mechanism
Derive **QKV**, **Swiglu** & **out projection** weights using the given input.

### How it works
1. Produce
- Has 3 trainable linear layers. Original (`w_O`), adjust (`w_A`) & transform weights (`w_T`).
- Original weights acts similar to Query weights in attention mechanism, it tells the model about the information original input (`X`) had.
- Adjust weights tell the model how to adjust the information presented by `Xw_O`.
- Transform weights tell the model to transform the adjusted information in a way which can be used by attention and mini-swiglu in attention.
- Original & Adjust weights shares the same shape `(C, C)`; where `C` is embedding dimension of the model.
- Transform weights has a shape `(C, 5*D + C)`; where `D` is QKV dimension.
- `(C, 5*D+C)` can be splitted into 3 weights; **w_qkv shape**: `(C, 3*D)`, **w_swiglu shape**: `(D, 2*C)` & **w_out shape**: `(C, C)`

2. Initialization
- We have **attn_w_qkv**, **attn_w_swiglu** & **attn_w_out**. QKV, Swiglu & out proj parameters of the attention mechanism.
- We also have **w_qkv**, **w_swiglu** & **w_out**. QKV, Swiglu & out proj parameters derived from the retention mechanism.
- We create 2 variables Current (`wC`) & Transformed (`wT`).
- Then we set them as the following `wC = tuple(attn_w_qkv, attn_w_swiglu, attn_w_out)` & `wT = tuple(w_qkv, w_swiglu, w_out)`.

3. Update rule
- First we always perform attention on `wC`.
- Then we update `wC` & `wT` in way given below:

```python
# [0] -> QKV; [1] -> Swiglu; [2] -> Output projection

wT, wC = wc, (
    norm(wT[0]) * F.silu(wC[0]) + norm(wC[0]),
    norm(wT[1]) * F.silu(wC[1]) + norm(wC[1]),
    norm(wT[2]) * F.silu(wC[2]) + norm(wC[2])
)
```

- Then we again perform the attention on new `wC` and this cycle continues.

## The Expert Abundance
MoE-powered attention mechanism & Swiglu MoE-FFN.

- Derive **QKV**, **Swiglu** & **Output projection** weights by the Retention Mechanism's **Update Rule**

```
Q, K, V     -> Local, Token-level MoE Scaled Dot Product Attention              -> Y
Y           -> Concatenate all local mixture of attention parts                 -> Y
Y           -> Token-level MoE Swiglu                                           -> Y
Y           -> X + out(Y)                                                       -> Y
```

## Citation

```
@software{Strawberry,
    author={Srijan Srivastava},
    title={Strawberry},
    url={https://github.com/SrijanSriv211/Strawberry},
    version={0.1.0},
    year = {2025}
}
```

<img src="img/rdr2.png" alt="lookwhosback" style="width:100%;">
