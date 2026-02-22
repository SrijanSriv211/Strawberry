# Strawberry: Is strawberry a fruit or a vegetable?

<img src="img/strawberry.png" alt="howmanyrsinthewordstrawberry" style="width:100%;">

Strawberry is primarily an early-stage neural network architecture built on top of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project. Currently not much is implemented, however everything inside this repository is enough to train AI models of various sizes.

Strawberry brings several improvements over the standard GPT-2 architecture, such as:
1. Shared stack of layers across recursion steps inspired from Google's Mixture of Recursions [[paper](https://arxiv.org/pdf/2507.10524)]
2. Use of Fourier series equation in attention inspired from this [[video](https://youtu.be/TkwXa7Cvfr8?t=920)]
3. `N:1` ratio attention placement inspired from Kimi Linear [[paper](https://arxiv.org/pdf/2510.26692)]
4. MoE Swiglu FFN & attention mechanism [[paper](https://arxiv.org/pdf/1701.06538)]
5. Apple's Attention Free Transformer [[paper](https://arxiv.org/pdf/2105.14103)]
6. Swiglu based FFN [[paper](https://arxiv.org/pdf/2002.05202)]
7. Modernized architecture: Rotary embeddings and QK-Norm
8. My custom `The Expert Abundance` attention mechanism
9. My custom `Retention Mechanism` architecture
10. Shared embedding weights

### Architecture Design
```
Input tokens
    |
[Token Embedding]
    |
[Strawberry Block, each containing:]
    [Retention steps ×N]
    |    |--- Retention Mechanism
    |    |--- The Expert Abundance Attention:
    |    |    |--- QKV projection
    |    |    |--- Apple's Attention Free Transformer (for 3 continuous retention steps, global context)
    |    |    |--- Scaled Dot Product Attention (every 4th retention step, local context, mixture of experts)
    |    |    |    |--- Rotary Positional Embeddings
    |    |    |    |--- QK Norm
    |    |    |    |--- Multi-Headed Attention
    |    |    |    |--- SiLU non-linearity
    |--- Swiglu FFN
    |
[Output Projection (weight-tied)]
    |
Next token logits
```

- Attention alternates between **AFT** and **SPDA**.
- AFT (global context) is applied for 3 continuous steps.
- SPDA (local context MoE) is then applied every 4th step.
- Giving a `3:1` AFT-to-SPDA ratio. This design is inspired by **Kimi Linear's** `N:1` KDA-to-MLA ratio.
- After `r_layer` updates, one final pass applies the learned **Swiglu FFN**.

## The Expert Abundance
MoE-attention mechanism & Swiglu mini-FFN.

> [!NOTE]
> As of now Token-level Local-Context-MoE has not been implemented in The Expert Abundance.

## Retention Mechanism
1. Calculate a low-rank linear-attention style 2D matrix using input (`x` of shape `(B, T, C)`).
2. Normalize it between `[-π, π]` and apply Fourier series.
3. Split it into 2 weights `w_qkv` & `w_out`.
4. Normalize it properly.

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
