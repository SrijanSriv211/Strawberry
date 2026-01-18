# Strawberry: Is strawberry a fruit or a vegetable?
- Strawberry is a model architecture.
- It brings several improvements over the standard Transformer architecture, such as:
1. Reuse a shared stack of layers across recursion steps inspired from Google's Mixture of Recursions [[paper](https://arxiv.org/pdf/2507.10524)]
2. Use a novel attention mechanism which I call `Attention On Detail`
3. Modernized architecture: Rotary embeddings, QK-Norm, and ReLU²
4. SwiGLU in feed forward network. [[paper](https://arxiv.org/pdf/2002.05202)]
5. The Muon optimizer [[writeup](https://kellerjordan.github.io/posts/muon)] [[repo](https://github.com/KellerJordan/Muon)]

### Attention On Detail
- [Multi-Headed Causal Self-Attention (MHA)](https://arxiv.org/pdf/1706.03762)
- [Attention Free Transformer (AFT)](https://arxiv.org/pdf/2105.14103)
- [Linear Attention Mechanism (LAM)](https://arxiv.org/pdf/2007.14902)
- [Key-Value Transformer](https://arxiv.org/pdf/2305.19129)
- [Neural Attention](https://arxiv.org/pdf/2310.11398)
- [SwiGLU](https://arxiv.org/pdf/2002.05202)

Although this version of attention on detail will be different from the Palm's version.

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
