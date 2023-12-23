# EfficientDDP-4-Contrastive-Train
Optimizing the way of contrastive learning in PyTorch-DDP distributed multi-GPU training



## Where you can use it?

Your own contrastive learning pre-train or fine-tune, architectures like [CLIP](https://openai.com/blog/clip/), [ALIGN](https://arxiv.org/abs/2102.05918).

```python
```







## Project sturcture



## Simple theoretical analysis

### Why contrastive learning is slightly complex in DDP?



[Separable and Non-separable Loss in detail](https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8)



### How to design gather function consistent with the gradient backward behavior of global computation?



### 







## Acknowledgement

