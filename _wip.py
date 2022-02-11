import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch import load

from nilearn import image, plotting, datasets, surface
from nilearn.plotting import plot_stat_map, view_img, view_img_on_surf
from nilearn.regions import signals_to_img_labels
from nilearn.image import load_img, mean_img
from nilearn.input_data import NiftiMasker

from matplotlib import pyplot as plt 

audio_path = "/home/maelle/DataBase/stimuli/movie10/life"
net_path = "/home/maelle/Results/20210719extract_feat_MIST_ROI_lr0.01_200epochs/sub-01/MIST_ROI_SoundNetEncoding_conv_1000_ks5_lr10_wd0.01_decoupled_lrSch_2021-07-19-23-54-38.pt"

# data = load(net_path)
# net = data['net']

from models import soundnet_model as snd
pytorch_param_path = './sound8.pth'
soundnet = snd.SoundNet8_pytorch()

# a = torch.load(pytorch_param_path)
# for key, o in a.items():
#     print(key, o.shape)

soundnet.load_state_dict(torch.load(pytorch_param_path))

paramaters_name = [] 
for layer_weights in soundnet.state_dict() : 
    if layer_weights.find('weight') > -1 or layer_weights.find('bias') > -1 : 
        paramaters_name.append(layer_weights)

train_limit = "conv5"
for name, param in zip(paramaters_name, soundnet.parameters()):
    if name.find(train_limit)>-1 : 
        break
    param.requires_grad = False

for name, param in zip(paramaters_name, soundnet.parameters()):
    print(name, param.requires_grad)


#-------to disable grad to certain layers in our net--(previously in models/encoding_model.py)-----------
self.parameters_name = [] 
for layer_weights in self.soundnet.state_dict() : 
    if layer_weights.find('weight') > -1 or layer_weights.find('bias') > -1 : 
        self.parameters_name.append(layer_weights)

freeze the parameters of soundNet up to desired training layer 
finetuning = False
for name, param in zip(self.parameters_name, self.soundnet.parameters()):
    if train_start is not None : 
        if name.find(self.train_start)>-1 : 
            #finetuning = True
            break
    param.requires_grad = False

if finetuning :
    print("Finetuning - backbone will be optimized up until "+str(train_start)+" included.") 
else : 
    print("Transfer learning - backbone is fixed")

#------------------------------------------------------------------------------------------------------- 