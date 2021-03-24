
from random import sample
import numpy as np
from audio_utils import load_audio_by_bit
import librosa
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

def create_usable_audiofmri_datasets(data, tr, sr, name='data'):
    print("getting audio files for {}...".format(name))
    x = []
    for (audio_path, mri_path) in data :
        length = librosa.get_duration(filename = audio_path)
        audio_segment = load_audio_by_bit(audio_path, 0, length, bitSize = tr, sr = sr)
        x.append(audio_segment)
    print("getting fMRI files for {}...".format(name))  
    y = [np.load(mri_path)['X'] for (audio_path, mri_path) in data]
    print("done.")

    #resize matrix to have the same number of tr : 
    for i, (seg_wav, seg_fmri) in enumerate(zip(x, y)) : 
        min_len = min(len(seg_fmri), len(seg_wav))
        x[i] = seg_wav[:min_len]
        y[i] = seg_fmri[:min_len]
    
    return(x,y)