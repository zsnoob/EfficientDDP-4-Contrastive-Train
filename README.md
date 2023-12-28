# EfficientDDP-4-Contrastive-Train
Optimizing the way of contrastive learning in PyTorch-DDP(DistributedDataParallel) multi-GPU training

* Transforming similarity calculation from [global, global] to [local, global]
* Addressing gradient issues in distributed calculation
* Aligning ground truth positions of local similarity matrix



## Project structure

```python
EfficientDDP-4-Contrastive-Train/
│
├── ED4CT/
│   ├── __init__.py
│   ├── AllGather.py
│   └── LossFunc.py
│
├── .gitignore
├── examples.py # toy examples to facilitate your understanding
├── LICENSE
└── README.md
```



## Where you can use it?

Your own contrastive learning pre-train or fine-tune project, using DDP to accelerate, architectures like [CLIP](https://openai.com/blog/clip/), [ALIGN](https://arxiv.org/abs/2102.05918).



## How to use it?

```python
# examlpe for training CLIP-LIKE models

import torch
import torch.distributed as dist
from ED4CT import AllGather
from ED4CT.LossFunc import CrossEntropy 

# ...
# dist.init_process_group(
#     backend='nccl',
#     rank=rank,
#     world_size=world_size,
# )
#
# model = nn.parallel.DistributedDataParallel(model, device_id) 
# ...

# your definition of model.forward()
def forward():
    # ...preprocess before calculating similarity matrix, just like non-distributed training
    # Now we get two features to calculate similarity matrix: features1 and features2
    # features1 and features2 will be shaped in [local_batch_size, feature_size]
	
    # suppose you only use DDP in training
    if self.training and self.task_conf.n_gpus > 1:
        # get all tensor from all GPU 
        # the shape of all_feature is [global_batch_size, feature_size]
        all_feature1 = AllGather(feature1)
        all_feature2 = AllGather(feature2)

        # the shape of sim_matrix is [local_batch_size, global_batch_size]
        sim_matrix1 = temperature * torch.matmul(feature1, all_feature2.t())
        sim_matrix2 = temperature * torch.matmul(feature2, all_feature1.t())

        # get ground truth position (the diagonal line's 1st element) of local data
        # align local ground truth
        ground_truth_pos = self.task_conf.global_batch_size / self.task_conf.n_gpus * dist.get_rank()

        loss1 = CrossEntropy(sim_matrix1, ground_truth_pos)
        loss2 = CrossEntropy(sim_matrix2, ground_truth_pos)

        loss = (loss1 + loss2) / 2
        
        return loss
```



## Simple theoretical analysis

The content below is not necessary for project deployment, just for deeper discussing about the reasoning process and motivations.



### Why contrastive learning is slightly complex in DDP?

We mainly discuss about two things for this section:

* Separable and Non-separable Loss
* Pytorch's mechanism of distributed gradient calculation——gradient bucket



#### Separable and Non-separable Loss

[Separable and Non-separable Loss in detail](https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8)

可分LOSS的GPU本地计算结果，以every `sigle sample`为视角完成了完整的LOSS计算过程，最后利用bucket在 `require_grad=True` 的 `parameters` 上平均梯度即可得到`batch`为单位的LOSS；分布式场景下，不可分LOSS在每个GPU本地计算的结果并不能作为一次完整的LOSS计算，DDP的默认行为不与非分布式场景下等价；



#### gradient bucket



- **Backward Pass**: The `backward()` function is directly invoked on the loss `Tensor`, which is out of DDP’s control, and DDP uses autograd hooks registered at construction time to trigger gradients synchronizations. When one gradient becomes ready, its corresponding DDP hook on that grad accumulator will fire, and DDP will then mark that parameter gradient as ready for reduction. When gradients in one bucket are all ready, the `Reducer` kicks off an asynchronous `allreduce` on that bucket to calculate mean of gradients across all processes. When all buckets are ready, the `Reducer` will block waiting for all `allreduce` operations to finish. When this is done, averaged gradients are written to the `param.grad` field of all parameters. So after the backward pass, the grad field on the same corresponding parameter across different DDP processes should be the same.

![gradient_bucket](./gradient_bucket.png)



### How to design "gather" backward function consistent with the gradient backward behavior of global computation?





### Implementation: all_reduce vs. reduce_scatter?

Which is actually do trade-off between computation-cost and distributed-communication-cost





## Version history

```
**Version 0.1.0 (December 28, 2023)**
- Initial src

TODO:
- finish theoretical analysis
_ finish examples.py
```





## Acknowledgements

Heuristic thoughts and discussions from PyTorch forum and openai/CLIP issues and comments.

Motivated by https://github.com/openai/CLIP/issues/132#issuecomment-908004353, deeply grateful to @jongwook for his generous and patient responses regarding implementation details in CLIP training.

