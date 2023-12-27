'''
This example file will show three different aspects of the ED4CT package:
1. Differences in gather gradient in advance and waiting for the gradient bucket
2. Why assign ground_truth_pos to loss function explicitly is necessary
3. How to use this package in your project
'''


def test_one_gather(mode):
    import torch
    import torch.distributed as dist
    from torch import nn
    from ED4CT.LossFunc import CrossEntropy
    from ED4CT import AllGather
    import os

    # distributed settings
    rank = int(os.environ['RANK'])  # global rank in the world
    world_size = int(os.environ['WORLD_SIZE'])  # GPU number
    device_id = int(os.environ["LOCAL_RANK"])  # GPU local number per node
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
    )

    # the gather method used in https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
    def all_gather_default(tensor):
        world_size = torch.distributed.get_world_size()
        with torch.no_grad:
            tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(tensor_list, tensor)
        tensor_list[torch.distributed.get_rank()] = tensor
        tensor_list = torch.cat(tensor_list, dim=0)
        return tensor_list

    # A toy model for contrastive learning, features will be extracted from two inputs
    # there are two linear layers in this model to be used as feature extractor
    class ContrastiveModel(nn.Module):
        def __init__(self, input_size, feature_size, gather_mode='default'):
            super(ContrastiveModel, self).__init__()
            self.feature_extractor1 = nn.Linear(input_size, feature_size)
            self.feature_extractor2 = nn.Linear(input_size, feature_size)
            self.gather_mode = gather_mode
            self._init_params()

        def _init_params(self):
            # initialize the parameters of two linear layers to one
            nn.init.ones_(self.feature_extractor1.weight)
            nn.init.ones_(self.feature_extractor1.bias)
            nn.init.ones_(self.feature_extractor2.weight)
            nn.init.ones_(self.feature_extractor2.bias)

        def forward(self, x1, x2, args):
            # 对两个输入分别进行特征提取
            features1 = self.feature_extractor1(x1)
            features2 = self.feature_extractor2(x2)
            if self.gather_mode == 'default':
                features1_all = all_gather_default(features1)
                features2_all = all_gather_default(features2)
            elif self.gather_mode == 'ED4CT':
                features1_all = AllGather(features1, args)
                features2_all = AllGather(features2, args)
            else:
                raise ValueError("gather_mode must be 'default' or 'ED4CT'")


            loss1 = torch.matmul(features1, features2_all.t())
            loss2 = torch.matmul(features2, features1_all.t())
            loss_fun = CrossEntropy()

            loss1 = loss_fun(loss1)
            loss2 = loss_fun(loss2)
            # loss = loss_fun(torch.cat([features1, features2], dim=0), ground_truth)
            # print(loss)
            # print(torch.autograd.grad(loss1.mean(), features2))
            # return loss
            # print(torch.autograd.grad(((loss1 + loss2) / 2).mean(), features1))
            return (loss1 + loss2) / 2

    class arg:
        def __init__(self, rank, size):
            self.rank = rank
            self.world_size = size

    args = arg(rank, world_size)

    model = ContrastiveModel(input_size=4, feature_size=3, gather_mode=mode).to(device_id)
    model1 = nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)

    torch.manual_seed(rank)
    x1 = torch.randn(1, 4).to(device_id)
    x2 = torch.randn(1, 4).to(device_id)

    loss = model1(x1, x2, args)

    loss.mean().backward()
    print("extractor1", str(model1.module.feature_extractor1.weight.grad[0]))
    # print("extractor2", str(model1.module.feature_extractor2.weight.grad[0]))
    # print(features1.grad_fn)
    # print(features1.grad_fn.next_functions)
    # print(features1.grad_fn.next_functions[0][0].next_functions)
    # print(features1.grad_fn.next_functions[0][0].variable)


def test_two_loss():
    import torch


    t = torch.tensor([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12], [13, 14, 15, 16, 17, 18]], dtype=torch.float32)
    '''
    similarity matrix will be shaped in [local_batch_size, global_batch_size]
    suppose we use two GPU and local_batch_size is three
    
    1  2  3  7  8  9
    4  5  6  10 11 12
    13 14 15 16 17 18
    '''

    # print(torch.diag(t)) tensor([ 1.,  5., 15.])

    # print(torch.diag(t, 0)) tensor([ 1.,  5., 15.])

    # print(torch.diag(t, 1)) tensor([ 2.,  6., 16.]), the diagonal element from [0, 1] to [2, 3]

    # print(torch.diag(t, 3)) tensor([ 7., 11., 18.]), the diagonal element from [0, 3] to [2, 5]

    # in practice, loss function will execute like that:

    local_batch_size_per_GPU = 3

    # suppose we run this code on the first GPU
    rank = 0
    # ground_truth_pos means the begin position of diagonal line in 1st row
    ground_truth_pos = local_batch_size_per_GPU * rank
    ground_truth_0th = torch.diag(t, ground_truth_pos)

    # ...in second GPU
    rank = 1
    ground_truth_pos = local_batch_size_per_GPU * rank
    ground_truth_1st = torch.diag(t, ground_truth_pos)

    # the results of torch.diag() will be used in loss function
    # to confirm the ground truth position in contrastive learning
    print(ground_truth_0th, ground_truth_1st) # tensor([ 1.,  5., 15.]) tensor([ 7., 11., 18.])


def test_three_use():
    print("This test is not runnable, just for reference.")
    '''
    # an example of how to use this package in your project
    
    import torch
    import torch.distributed as dist
    from ED4CT import AllGather
    from ED4CT.LossFunc import CrossEntropy 
    
    
    # your definition of model.forward()
    def forward():
        # ...preprocess before calculating similarity matrix, just like non-distributed training
        # Now we get two features to calculate similarity matrix: features1 and features2
        # features1 and features2 will be shaped in [local_batch_size, feature_size]
        
        if self.training and self.task_conf.n_gpus > 1:
            # get all tensor from all GPU 
            # the shape of all_feature is [global_batch_size, feature_size]
            all_feature1 = AllGather(feature1)
            all_feature2 = AllGather(feature2)
            
            # the shape of sim_matrix is [local_batch_size, global_batch_size]
            sim_matrix1 = temperature * torch.matmul(feature1, all_feature2.t())
            sim_matrix2 = temperature * torch.matmul(feature2, all_feature1.t())
            
            # get ground truth position of local data
            ground_truth_pos = self.task_conf.global_batch_size / self.task_conf.n_gpus * dist.get_rank()
            
            loss1 = CrossEntropy(sim_matrix1, ground_truth_pos)
            loss2 = CrossEntropy(sim_matrix2, ground_truth_pos)
            
            loss = (loss1 + loss2) / 2
    
    '''


if __name__ == "__main__":
    # choose one of the following functions to run, comment out the others
    # for test_one_gather(), you should run this file with command: torchrun --nproc_per_node=2 examples.py
    # advice: use two GPU is enough to run example one
    # two gather mode to test: 'default' and 'ED4CT'

    # test_one_gather('default')
    # test_one_gather('ED4ct')
    # test_two_loss()
    # test_three_use()
    pass