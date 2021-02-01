#Generic
import os
import warnings
import numpy as np
from datetime import datetime
#Get input & preprocess
from files_utils import associate_stimuli_with_Parcellation, create_dir_if_needed
from main_model_training import main_model_training
from models import encoding_models as encod
from torch import nn, optim
from train_utils import EarlyStopping


date = datetime.now()
dt_string = date.strftime("%Y%m%d")
#-------------------------------ARGUMENTS----------------------------------------------
#data selection
bourne = 'bourne_supremacy'
wolf = 'wolf_of_wall_street'
life = "life"
hidden = "hidden_figures"

films = [bourne, wolf, life, hidden]
subjects = [0,1,2,3,4]

data_processing = {
    'scale':'roi',                                      #'voxel',
    'tr' : 1.49,
    'sr' : 22050,
    'selected_ROI':[141,152,153,169,170,204,205],       #None,
    'nroi': 7   #210                                    #556
}

#model parameters
allfmrihidden=[1000]
models = [encod.SoundNetEncoding_conv]
kernel_sizes=[1, 5, 10, 15]

training_hyperparameters = {
    'gpu':False,
    'batchsize':30,
    'lr':0.01,
    'nbepoch': 500,
    'train_percent':0.6,
    'test_percent':0.2,
    'val_percent':0.2,
    'mseloss':nn.MSELoss(reduction='sum'),
    #'early_stopping':EarlyStopping(patience=10, verbose=True,delta=1e-6)
    #problem, that stop the training of any following test
    #to look how to implement a more interactable early_stopping
}

#paths
outpath = "/home/maelle/Results/"
stimuli_path = '/home/maelle/Database/cneuromod/movie10/stimuli' #'/home/brain/Data_Base/cneuromod/movie10/stimuli' 
path_parcellation = '/home/maelle/Database/12_2020_parcellation/auditory_Voxels/20210115_NORMALIZED' #/home/maelle/Database/movie10_parc';'/home/brain/Data_Base/movie10_parc'
all_subs_files = associate_stimuli_with_Parcellation(stimuli_path, path_parcellation)
resultpath = outpath+dt_string+"_tests_roi_kernel_{}Norm_embed2020".format(data_processing['scale'])
create_dir_if_needed(outpath)

#--------------------------TRAINING LOOP-------------------------------------------------------------------
if __name__ == "__main__":
    for subject in subjects:
        outpath_sub = os.path.join(resultpath, 'subject_'+str(subject))
        create_dir_if_needed(outpath_sub)

        for film in films:
            outpath_film = os.path.join(outpath_sub, film)
            create_dir_if_needed(outpath_film)

            for model in models:
                for fmrihidden in allfmrihidden:
                    for kernel_size in kernel_sizes:
                        
                        data_selection = {
                            'all_data':all_subs_files,
                            'subject':subject,
                            'film':film
                        }

                        training_hyperparameters['model'] = model
                        training_hyperparameters['fmrihidden'] = fmrihidden
                        training_hyperparameters['kernel_size'] = kernel_size

                        main_model_training(outpath_film, data_selection, data_processing, training_hyperparameters)
            

                



