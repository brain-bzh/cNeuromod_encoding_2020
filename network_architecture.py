import os
from torchinfo import summary
import torch
from torch.utils.data import DataLoader

import files_utils as fu
from models import encoding_models as encod
from Datasets_utils import SequentialDataset, create_usable_audiofmri_datasets, create_train_eval_dataset

sub=  '03'

nInputs= 556 #210
scale = 'auditory_Voxels' #'MIST_ROI' 
output_layer='conv7'
kernel_size = 5
finetune_start = None
finetune_delay = 0
tr=1.49
sr=22050
batchsize=70

outpath = "/home/maellef/Results/test_architecture"
stimuli_path = '/home/maellef/DataBase/stimuli/'
embed_path = '/home/maellef/DataBase/fMRI_Embeddings'

dataset_path = os.path.join(stimuli_path, 'movie10')
parcellation_path = os.path.join(embed_path, scale, 'movie10', 'sub-'+sub)

train_dataset=['wolf']
eval_dataset=['wolf']
all_subs_files = dict()

for film in os.listdir(dataset_path):
    film_path = os.path.join(dataset_path, film)
    if os.path.isdir(film_path):
        all_subs_files[film] = fu.associate_stimuli_with_Parcellation(film_path, parcellation_path)

train_data = []
for subdata in train_dataset:
    train_data.extend(all_subs_files[subdata]) 
eval_data = []
for subdata in eval_dataset:
    eval_data.extend(all_subs_files[subdata])

train_input = [(data[0], data[1]) for data in train_data]
eval_input = [(data[0], data[1]) for data in eval_data]

DataTrain, DataVal, DataTest = create_train_eval_dataset(train_input, eval_input, 
                                                         train_percent=0.6, 
                                                         val_percent=0.2, 
                                                         test_percent=0.2)

xTrain, yTrain = create_usable_audiofmri_datasets(DataTrain, tr, sr, name='training')
TrainDataset = SequentialDataset(xTrain, yTrain, batch_size=batchsize)
trainloader = DataLoader(TrainDataset, batch_size=None)

net = encod.SoundNetEncoding_conv(pytorch_param_path="./sound8.pth",out_size=nInputs, 
                                    output_layer=output_layer, kernel_size=kernel_size, 
                                    train_start= finetune_start, finetune_delay=finetune_delay)
a = iter(trainloader)
input, output = next(a)
input, output = next(a)
input, output = next(a)
x =  torch.Tensor(input).view(1,1,-1, 1)
print(input.shape, x.shape) 
summary(net, input_data=[x, 0], col_width=20, depth=4, mode='train', 
        col_names=("input_size","output_size", "num_params",
                   "kernel_size", "mult_adds"))
print(summary)
