import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


class CrossEntropy(nn.Module):
    def __init__(self,):
        super(CrossEntropy, self).__init__()

    # the shape of sim_matrix: [local_batch_size, global_batch_size]
    def forward(self, sim_matrix, ground_truth_pos=0):
        logpt = F.log_softmax(sim_matrix, dim=-1)

        # Note that different GPU must use different ground_truth_position
        logpt = torch.diag(logpt, ground_truth_pos)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss