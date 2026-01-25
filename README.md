# Strawberry: Is strawberry a fruit or a vegetable?
- Strawberry is a model architecture.
- It is built on top of [Palm](https://github.com/SrijanSriv211/Palm)
- It brings several improvements over the standard Transformer architecture, such as:
    1. Shared stack of layers across recursion steps inspired from Google's Mixture of Recursions [[paper](https://arxiv.org/pdf/2507.10524)]
    2. Modernized architecture: Rotary embeddings and QK-Norm
    3. Shared embedding weights
    4. MoE based attention mechanism `The Expert Abundance`
    5. `Compaction Compute` layer makes the entire architecture independent of batch-size and context-length
- `Strawberry s1` language model based on this architecutre. **COMING SOON**

## The Expert Abundance

`The Expert Abundance` is a newer version of `Attention On Detail` which I introduced in my previous project `Palm`.

- [Multi-Headed Causal Self-Attention (MHA)](https://arxiv.org/pdf/1706.03762)
- [Attention Free Transformer (AFT)](https://arxiv.org/pdf/2105.14103)
- [SwiGLU](https://arxiv.org/pdf/2002.05202)

```
X           -> Linear                                               -> QKV
Q, K        -> Rotary embeddings                                    -> Q, K
Q, K, V     -> Full/Global AFT                                      -> V'
Q, K, V'    -> Local, Mixture of Scaled Dot Product Attention       -> Y
Y           -> Concatenate all local mixture of attention parts     -> Y
Y           -> Swiglu                                               -> Y
Y           -> X + Y                                                -> Y
```

## Compaction Compute

Compaction computation on an input makes it batch-size & block-size independent.

> [!NOTE]
> Compaction Compute layer is currently untested feature. So take it with salt. A lot of salt.

### How it works?

It looks quite similar to the standard attention mechanism formula.

1. `WO` (Weights Original), `WA` (Weights Adjust), `WT` (Weights Transform) are 3 linear layers of dimensions `(C, C)` each, where `C` is the embedding dimension of the model.

2. Pass an `RMS_norm(x)` into WO and WA to get O and A.

3. Our input `x` right now has a shape of `(B, T, C)` where `B = batch size`, `T = context length`, `C = embedding dimension`.

4. Similarly O & A also has a shape of `(B, T, C)`.

5. Reshape them both into a shape of `(B*T, C)`.

6. Therefore, O.T (where `.T` implies Transpose) will have a shape of `(C, B*T)`

7. Now take a scaled dot product between `O.T` & `A` and normalize it with square root of `1/C`.

8. Then pass that into `pi*tanh` function and take a dot product of that with `WT`.

9. Proper formula is the following: `y = pi * tanh((O.T @ A) / sqrt(embd_dim)) @ T`.

10. It will give us `y` with a shape of `(C, C)`

11. We have now compacted our input of shape `(B, T, C)` to `(C, C)` and made it completely independent of batch-size and block-size.

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
