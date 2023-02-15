import torch
from torch.utils.data import *
# from torch.utils.data import *

class Batchsampler_testv100(BatchSampler):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)
        
    def __iter__(self):
        batch_fix = []
        for idx in self.sampler:
            batch_fix.append(idx)
            if len(batch_fix) == self.batch_size:
                break
        batch = []
        for i, idx in enumerate(self.sampler):
            batch.append(batch_fix[i])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            
        if len(batch) > 0 and not self.drop_last:
            yield batch
            
        return 
        
class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source) -> None:
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)
        