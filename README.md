# Strawberry: Is strawberry a fruit or a vegetable?

<img src="img/strawberry.png" alt="howmanyrsinthewordstrawberry" style="width:100%;">

Strawberry is primarily an early-stage experimental neural network architecture built on top of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project. Goal of Strawberry is to experiment with different & novel ideas as much, as fast and in as different ways as possible.

Changes over the standard nanoGPT architecture:
1. Modernized architecture: Rotary embeddings and QK-Norm
2. Silia (Silu in Attention), replace FFN with attention
3. Attention residuals proposed by MoonshotAI ([paper](https://arxiv.org/pdf/2603.15031))
4. Applied Qwen3-Next's gated attention ([paper](https://arxiv.org/pdf/2505.06708))
5. Shared embedding weights

### Architecture Design
```
Input tokens
    |
[Token Embedding]
    |
[Strawberry Block xN:]
    |--- Multi-Headed Attention
    |    |--- Rotary Positional Embeddings
    |    |--- QK Norm
    |    |--- Scaled Dot Product Attention
    |--- Silu activation function
    |--- Multi-Headed Attention
    |--- Attention Residuals
[Output Projection (weight-tied)]
    |
Next token logits
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
