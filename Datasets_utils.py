
from random import sample
from torch.utils.data import IterableDataset
import torch

class SequentialDataset(IterableDataset):
    def __init__(self, x, y, batch_size):
        super(SequentialDataset).__init__()

        self.x = x
        self.y = y
        self.batch_size = batch_size

        self.batches = []
        for seg_x, seg_y in zip(x,y):
            seg = self.__create_batchs__(seg_x, seg_y)
            self.batches.extend(seg)

        self.batches = sample(self.batches, len(self.batches))
        print(len(self.batches))

    def __create_batchs__(self, dataset_x, dataset_y):
        batches = []
        total_nb_inputs = len(dataset_x)
        for batch_start in range(0, total_nb_inputs, self.batch_size):
            if batch_start+self.batch_size>total_nb_inputs:
                batch_end = total_nb_inputs
            else:
                batch_end = batch_start+self.batch_size

            batches.append((torch.tensor(dataset_x[batch_start:batch_end]), torch.tensor(dataset_y[batch_start:batch_end])))
        batches = sample(batches, len(batches))
        return batches

    def __len__(self):
        return(len(self.batches)) 

    def __iter__(self):
        return iter(self.batches)