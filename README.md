# EfficientDDP-4-Contrastive-Train
Optimizing the way of contrastive learning in PyTorch-DDP distributed multi-GPU training

* Transforming similarity calculation from [global, global] to [local, global]
* Addressing gradient issues in distributed calculation
* Aligning ground truth positions of local similarity matrix



## Project structure

```
```



## Where you can use it?

Your own contrastive learning pre-train or fine-tune project, architectures like [CLIP](https://openai.com/blog/clip/), [ALIGN](https://arxiv.org/abs/2102.05918).

```python
# examlpe for training CLIP-LIKE models

import torch
import torch.distributed as dist
from ED4CT import loss_fun



```





## 





## Simple theoretical analysis

The content below is not necessary for project deployment, just for deeper discussing about the reasoning process and motivations.

### Why contrastive learning is slightly complex in DDP?

We mainly discuss about two things for this section:

* Separable and Non-separable Loss
* Pytorch's mechanism of distributed gradient calculation——gradient bucket



[Separable and Non-separable Loss in detail](https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8)

可分LOSS的GPU本地计算结果，以every `sigle sample`为视角完成了完整的LOSS计算过程，最后利用bucket在 `require_grad=True` 的 `parameters` 上平均梯度即可得到`batch`为单位的LOSS；分布式场景下，不可分LOSS在每个GPU本地计算的结果并不能作为一次完整的LOSS计算，DDP的默认行为不与非分布式场景下等价；

### How to design "gather" backward function consistent with the gradient backward behavior of global computation?





### Implementation: all_reduce vs. reduce_scatter?

Which is actually do trade-off between computation-cost and distributed-communication-cost





## Acknowledgements

Pytorch forum and OPENAI/CLIP issues and comments.
