import os
import numpy as np
from torch import load, device, save
import torch

from models import *
from models.encoding_models import SoundNetEncoding_conv


def convert_model(model_file_path, savepath,device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    modeldict = load(model_file_path, map_location=device)
    
    model = modeldict['model']
    
    
    hp = modeldict['hyperparameters']
    print(hp)

    newdict={}

    newdict['out_size']=model.out_size
    newdict['output_layer']=model.output_layer
    newdict['fmrihidden']=model.fmrihidden
    newdict['kernel_size']=hp['kernel_size']
    newdict['power_transform']=model.power_transform
    newdict['checkpoint'] = modeldict['net']

    # Define model 

    newmodel = SoundNetEncoding_conv(out_size=newdict['out_size'],output_layer=newdict['output_layer'],
    kernel_size=newdict['kernel_size'])
    newmodel.load_state_dict(newdict['checkpoint'])

    torch.save(newdict,savepath)

    return newmodel