#Generic
import os, argparse
import numpy as np
from datetime import datetime
#Get input & preprocess
import files_utils as fu
#Create Dataset
from torch.utils.data import DataLoader
from Datasets_utils import SequentialDataset, create_usable_audiofmri_datasets, create_train_eval_dataset
#Models
from models import encoding_models as encod
#Train & Test
from tqdm import tqdm
from torch import nn, optim, save, load
from train_utils import train, test, test_r2, EarlyStopping 

parser = argparse.ArgumentParser()

#data_selection
parser.add_argument("-s", "--sub", type=str, nargs='+')
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("--sessionsTrain", type=int, default=1) # WIP, must be >=1, add a condition to check the entry
parser.add_argument("--sessionsEval", type=int, default=1) # WIP, must be >=1, add a condition to check the entry
parser.add_argument("--trainData", type=str, nargs='+')
parser.add_argument("--evalData", type=str, nargs='+')

#data_processing
parser.add_argument("--scale", type=str)
parser.add_argument("--select", type=int, nargs='+', default=None) # in case we want to learn on specific ROI/Voxels
parser.add_argument("--tr", type=float, default=1.49)
parser.add_argument("--sr", type=int, default=22050)

args = parser.parse_args()

data_selection = {
    'subject' : args.sub,
    'dataset' : args.dataset,
    'train_data' : args.trainData,
    'eval_data' : args.evalData,
    'sessions_train' : args.sessionsTrain,
    'sessions_eval' : args.sessionsEval
}
ds = data_selection

data_processing = {
    'scale': args.scale,
    'tr' : args.tr,
    'sr' : args.sr,
    'selected_inputs': args.select                     
}
dp = data_processing

outpath = "/home/maellef/Results/finefriends_groupmodel" #'/home/maellef/scratch/Results/'
stimuli_path = '/home/maellef/DataBase' #'/home/maellef/projects/def-pbellec/maellef/data/stimuli'
embed_path = '/home/maellef/DataBase/fMRI_Embeddings' #'/home/maellef/projects/def-pbellec/maellef/data/fMRI_Embeddings_fmriprep-2022'

dataset_path = os.path.join(stimuli_path, ds['dataset'], 'stimuli')
parcellation_path = os.path.join(embed_path, dp['scale'], ds['dataset'])
                                 
                                 #'sub-'+args.sub)

all_subs_files = dict()
for sub in ds['subject']:
    print(sub)
    for film in os.listdir(dataset_path):
        if film not in all_subs_files.keys():
            all_subs_files[film] = []
        film_path = os.path.join(dataset_path, film)
        if os.path.isdir(film_path):
            sub_parcellation_path = os.path.join(parcellation_path, 'sub-'+sub)
            all_subs_files[film].extend(fu.associate_stimuli_with_Parcellation(film_path, sub_parcellation_path))

resultpath = os.path.join(outpath, dt_string+"test")
resultpath = os.path.join(resultpath, 'sub-'+args.sub)

ds['all_data']=all_subs_files

python model_training.py -s 02 03 04 05 06 -d friends --trainData s1 s2 s3 --evalData s4 --scale auditory_Voxels -f conv4 --bs 75 --ks 7 --lr 1e-5 --wd 1e-3 --patience 15 --delta 0 --lrScheduler --decoupledWD
python model_training.py -s 02 03 04 05 06 -d friends --trainData s1 s2 s3 --evalData s4 --scale MIST_ROI -f conv4 --bs 75 --ks 7 --lr 1e-5 --wd 1e-3 --patience 15 --delta 0 --lrScheduler --decoupledWD 