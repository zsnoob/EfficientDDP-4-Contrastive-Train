__version__ = "0.2.0"

from .AllGather import AllGather
from .LossFunc import CrossEntropy


__all__ = [ '__version__',
            'AllGather',
            'CrossEntropy', # you can add more functions here
]

print("ED4CT warning:\nIf you run your project on multi-GPU, you should assign ground_truth_pos to loss function explicitly.")