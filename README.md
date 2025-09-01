# Strawberry: Is strawberry a fruit or a vegetable?
- It's based on my another AI project, [Palm](https://github.com/SrijanSriv211/Palm).
- Strawberry is an attempt to make Palm better and make a good AI model which feels human-like.
- The first AI model under this name `Strawberry` is going to be `Strawberry s1`.

## Palm is a tree, not a language model.
Since Strawberry is based on Palm, it built on top of same improvements.
- Reuse a shared stack of layers across recursion steps inspired from Google's Mixture of Recursions [[paper](https://arxiv.org/pdf/2507.10524)]
- Use a novel attention mechanism which I call `Attention On Detail`
- Modernized architecture: Rotary embeddings, QK-Norm, and ReLU²
- Parallel layers proposed by Google's PaLM [[paper](https://arxiv.org/pdf/2204.02311)]
- SwiGLU in feed forward network. [[paper](https://arxiv.org/pdf/2002.05202)]
- Untie head from embedding
- The Muon optimizer [[writeup](https://kellerjordan.github.io/posts/muon)] [[repo](https://github.com/KellerJordan/Muon)]
- Linear layer factorization

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
