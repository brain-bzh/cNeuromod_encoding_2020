import os, argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim, save, load
import files_utils as fu
from train_utils import test_r2
from Datasets_utils import SequentialDataset, create_usable_audiofmri_datasets, create_train_eval_dataset
from models import encoding_models as encod

models_path = '/home/maellef/Results/best_models/converted' #'/home/maellef/projects/def-pbellec/maellef/best_models/best_models' 
dataset_path = '/home/maellef/DataBase/fMRI_Embeddings' #'/home/maellef/projects/def-pbellec/maellef/data/fMRI_Embeddings_fmriprep-2022/'
stimuli_path = '/home/maellef/DataBase/stimuli'    #movie10/wolf' #'/home/maellef/projects/def-pbellec/maellef/data/stimuli/friends/s04/'
outpath = '/home/maellef/Results'
save_path = os.path.join(outpath, 'one_run_eval')
os.makedirs(save_path, exist_ok=True) 
#subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
#sub_model = 'sub-01'
#datasets = ['friends']
#scales = ['auditory_Voxels', 'MIST_ROI']

def load_sub_models(sub, scale, conv, models_path=models_path, no_init=False): 
    models = {}
    #scale_path = os.path.join(models_path, sub, scale)
    for model in os.listdir(models_path):
        if '.pt' in model and conv in model and sub in model and scale in model:
            model_path = os.path.join(models_path, model)
            modeldict = torch.load(model_path, map_location=torch.device('cpu'))
            model_net = encod.SoundNetEncoding_conv(out_size=modeldict['out_size'],output_layer=modeldict['output_layer'],
                                                    kernel_size=modeldict['kernel_size'], no_init=no_init)
            if not no_init:
                model_net.load_state_dict(modeldict['checkpoint'])
            models[model] = model_net
            if no_init:
                break
    return models
 
def one_run_eval(sub, dataset, models_dict, pairs_wav_fmri, no_init=False, outpath = save_path):
    shape = 210 if scale == 'MIST_ROI' else 556
    for name, model in models_dict.items():
        all_runs = np.array([]).reshape(-1,shape)
        batchsize = int(name[name.find('conv_')+len('conv_'):name.find('conv_')+len('conv_')+3])
        print('batchsize '+str(batchsize))
        for pair in pairs_wav_fmri:
            eval_input = [pair] if len(pair) == 2 else [(pair[0], pair[1])]
            xTest, yTest = create_usable_audiofmri_datasets(eval_input, tr=1.49, sr=22050, name='test')
            TestDataset = SequentialDataset(xTest, yTest, batch_size=batchsize, selection=None)
            testloader = DataLoader(TestDataset, batch_size=None)    
            r2_score = test_r2(testloader, net=model, epoch=1, mseloss=nn.MSELoss(reduction='sum'), gpu=False)
            all_runs = np.append(all_runs, r2_score.reshape(1,-1), axis=0)
        print(all_runs.shape)

        #sub_path = os.path.join(save_path, dataset, sub)
        #sub_path = sub_path if not no_init else os.path.join(sub_path, 'no_init_model')
        #os.makedirs(sub_path, exist_ok=True)
        sub_path = outpath
        savepath = os.path.join(sub_path, name[:-3])
        np.save(savepath, all_runs)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-d", "--dataset", type=str)
    #parser.add_argument("-s", "--scale", type=str)
    #parser.add_argument('-c', '--conv', type=str)
    #parser.add_argument("-S", "--sub_data", type=str)
    #parser.add_argument("-M", "--sub_model", type=str)
    #parser.add_argument("--noInit", dest='noInit', action='store_true')
    #
    #args = parser.parse_args()
    #dataset = args.dataset
    #scale = args.scale
    #conv = args.conv
    #sub_data = args.sub_data
    #sub_model = args.sub_model
    #no_init = args.noInit

    no_init = False
    conv = 'opt110_wb' #'conv4'
    for scale in ['auditory_Voxels', 'MIST_ROI']:
        for sub_model in ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']:
            sub_data = sub_model 
            for sub_dataset in ['s04', 'bourne', 'figures', 'life', 'wolf']:
                dataset = 'friends' if sub_dataset == 's04' else 'movie10'
                result_path = os.path.join(save_path, dataset, sub_dataset)
                data_stimuli_path = os.path.join(stimuli_path, dataset, sub_dataset)
                os.makedirs(result_path, exist_ok=True) 

                print(conv, scale, sub_model, dataset, sub_dataset)

                models = load_sub_models(sub_model, scale, conv, no_init=no_init)
                parcellation_path = os.path.join(dataset_path, scale, dataset, sub_data)
                pairs_wav_fmri = fu.associate_stimuli_with_Parcellation(data_stimuli_path, parcellation_path)
                one_run_eval(sub_data, dataset, models, pairs_wav_fmri, no_init=no_init, outpath=result_path)

#python one_run_eval.py -d friends -s auditory_Voxels -c conv1 --sub_model sub-01 --sub_data sub-01
#create a dataset for eval made of one run




