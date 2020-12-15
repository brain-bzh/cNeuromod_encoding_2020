
from random import sample
from torch.utils.data import IterableDataset
import torch

class SequentialDataset(IterableDataset):
    def __init__(self, x, y, batch_size, selection = None):
        super(SequentialDataset).__init__()

        self.x = x
        self.y = self.__select_Y_output__(y, selection)
        self.batch_size = batch_size

        self.batches = []
        for seg_x, seg_y in zip(self.x,self.y):
            seg = self.__create_batchs__(seg_x, seg_y)
            self.batches.extend(seg)

        self.batches = sample(self.batches, len(self.batches))
    
    def __select_Y_output__(self, dataset_y, selection) : 
        if selection == None:
            return dataset_y

        selected_y = [] 
        for seg in dataset_y:
            new_seg = []
            for tr in seg:
                roi_selection = [tr[index] for index in selection]
                new_seg.append(roi_selection)
            selected_y.append(new_seg)

        return selected_y

    def __create_batchs__(self, dataset_x, dataset_y):
        batches = []
        total_nb_inputs = len(dataset_x)
        for batch_start in range(0, total_nb_inputs, self.batch_size):
            if batch_start+self.batch_size>total_nb_inputs:
                batch_end = total_nb_inputs
            else:
                batch_end = batch_start+self.batch_size

            batches.append((torch.Tensor(dataset_x[batch_start:batch_end]), torch.Tensor(dataset_y[batch_start:batch_end])))
        batches = sample(batches, len(batches))
        return batches

    def __len__(self):
        return(len(self.batches)) 

    def __iter__(self):
        return iter(self.batches)