# Strawberry: Is strawberry a fruit or a vegetable?
- Strawberry is a model architecture.
- It is built on top of [Palm](https://github.com/SrijanSriv211/Palm)
- It brings several improvements over the standard Transformer architecture, such as:
    1. Share stack of layers across recursion steps inspired from Google's Mixture of Recursions [[paper](https://arxiv.org/pdf/2507.10524)]
    2. Modernized architecture: Rotary embeddings and QK-Norm
    3. MoE based attention mechanism. `The Expert Abundance`

## The Expert Abundance

`The Expert Abundance` is a newer version of `Attention On Detail` which I introduced in my previous project `Palm`.

- [Multi-Headed Causal Self-Attention (MHA)](https://arxiv.org/pdf/1706.03762)
- [Attention Free Transformer (AFT)](https://arxiv.org/pdf/2105.14103)
- [SwiGLU](https://arxiv.org/pdf/2002.05202)
- [MoE](https://arxiv.org/pdf/2507.11181)

```
X           -> Linear                                               -> QKV
Q, K        -> Rotary embeddings                                    -> Q, K
Q, K, V     -> Full/Global AFT                                      -> Y
Y           -> Linear                                               -> QKV
Q, K        -> Rotary embeddings                                    -> Q, K
Q, K, V     -> Local, Mixture of Scaled Dot Product Attention       -> Y
Y           -> Concatenate all local mixture of attention parts     -> Y
Y           -> Swiglu                                               -> Y
Y           -> X + Y                                                -> Y
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
