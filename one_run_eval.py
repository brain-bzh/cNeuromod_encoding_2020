import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim, save, load
import files_utils as fu
from train_utils import test_r2
from Datasets_utils import SequentialDataset, create_usable_audiofmri_datasets, create_train_eval_dataset
from models import encoding_models as encod

models_path = '/home/maellef/projects/def-pbellec/maellef/best_models' 
dataset_path = '/home/maellef/projects/def-pbellec/maellef/data/DataBase/fMRI_Embeddings_fmriprep-2022/'
stimuli_path = '/home/maellef/projects/def-pbellec/maellef/data/DataBase/stimuli/friends/s04/'


subjects = ['sub-01']
datasets = ['friends']
#-----load the model----------------------------------------------------------------

for scale in ['MIST_ROI']: #os.listdir(models_path):
    models = {}
    scale_path = os.path.join(models_path, scale)
    for model in os.listdir(scale_path):
        if '.pt' in model:
            model_path = os.path.join(scale_path, model)
            modeldict = torch.load(model_path, map_location=torch.device('cpu'))
            model_net = encod.SoundNetEncoding_conv(out_size=modeldict['out_size'],output_layer=modeldict['output_layer'],kernel_size=modeldict['kernel_size'])
            model_net.load_state_dict(modeldict['checkpoint'])
            models[model] = model_net

    for dataset in datasets:
        for sub in subjects:
            print(sub)
            parcellation_path = os.path.join(dataset_path, scale, dataset, sub)
            pairs_wav_fmri = fu.associate_stimuli_with_Parcellation(stimuli_path, parcellation_path)
            for name, model in models.items():
                if sub in name:
                    print(name)
                    shape = 210 if scale == 'MIST_ROI' else 556
                    all_runs = np.array([]).reshape(-1,shape)
                    batchsize = int(name[name.find('conv_')+len('conv_'):name.find('conv_')+len('conv_')+3])
                    for pair in pairs_wav_fmri:
                        eval_input = [pair] if len(pair) == 2 else [(pair[0], pair[1])]
                        xTest, yTest = create_usable_audiofmri_datasets(eval_input, tr=1.49, sr=22050, name='test')
                        TestDataset = SequentialDataset(xTest, yTest, batch_size=batchsize, selection=None)
                        testloader = DataLoader(TestDataset, batch_size=None)
    
                        r2_score = test_r2(testloader, net=model, epoch=1, mseloss=nn.MSELoss(reduction='sum'), gpu=False)
                        all_runs = np.append(all_runs, r2_score.reshape(1,-1), axis=0)
                    print(all_runs.shape)
                    savepath = os.path.join(models_path, name[:-3])
                    np.save(savepath, all_runs)
    
#create a dataset for eval made of one run




