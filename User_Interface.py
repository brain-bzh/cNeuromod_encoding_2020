#Generic
import os
import warnings
import numpy as np
from datetime import datetime
#Get input & preprocess
import files_utils as fu
import visualisation_utils as vu
from main_model_training import main_model_training
from models import encoding_models as encod
from torch import nn, optim, load, device
from train_utils import EarlyStopping


date = datetime.now()
dt_string = date.strftime("%Y%m%d")
#-------------------------------ARGUMENTS----------------------------------------------
#data selection
bourne = 'bourne_supremacy'
wolf = 'wolf_of_wall_street'
life = "life"
hidden = "hidden_figures"

films = [bourne]#, wolf, life, hidden]
subjects = [0]#,1,2,3,4]

data_processing = {
    'scale': 'MIST_ROI', #'auditory_Voxels'
    'tr' : 1.49,
    'sr' : 22050,
    'selected_ROI':[141,152,153,169,170,204,205], #None
    'nroi': 7 #556   #210                               
}

#model parameters
allfmrihidden=[1000]
models = [encod.SoundNetEncoding_conv]
kernel_sizes=[0, 5, 10, 15]

training_hyperparameters = {
    'gpu':False,
    'batchsize':30,
    'lr':0.0001,
    'nbepoch': 60,
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
stimuli_path = '/home/maelle/Database/fMRI_datasets/cneuromod/movie10/stimuli' #'/home/brain/Data_Base/cneuromod/movie10/stimuli' 
path_embed = '/home/maelle/Database/Embeddings/cNeuromod/embed_2020'
embed_used = os.path.basename(path_embed)
path_parcellation = os.path.join(path_embed, data_processing['scale']) #'/home/brain/Data_Base/movie10_parc'
all_subs_files = fu.associate_stimuli_with_Parcellation(stimuli_path, path_parcellation)
resultpath = outpath+dt_string+"_tests_kernel_{}_{}_3".format(data_processing['scale'], embed_used)
fu.create_dir_if_needed(outpath)

#--------------------------TRAINING LOOP-------------------------------------------------------------------
if __name__ == "__main__":
    for subject in subjects:
        outpath_sub = os.path.join(resultpath, 'subject_'+str(subject))
        fu.create_dir_if_needed(outpath_sub)

        for film in films:
            outpath_film = os.path.join(outpath_sub, film)
            fu.create_dir_if_needed(outpath_film)

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

    fu.rename_object(resultpath, 'subject_', fu.cNeuromod_subject_convention, objects=['dirs'])
    #results visu

    out_directory = os.path.join(resultpath, 'analysis')
    fu.create_dir_if_needed(out_directory)
    auditorymask='STG_middle.nii.gz'

    all_data = vu.construct_data_dico(data_path=resultpath, extension='.pt', criterion='sub', target='ks')
    #all_maps = construct_data_dico('film', '.gz', target_path)
    all_data = vu.sort_by_target_value(all_data)
    for sub, films in all_data.items():
        all_loaded = [(dir_name, load(file_path, map_location=device('cpu')), target_data) for (dir_name, file_path, target_data) in films]
        all_data[sub] = all_loaded

    all_data = vu.subdivise_dict(all_data)

    measures = ["loss", "r2_max", "r2_mean"]
    for subject, data in all_data.items():
        vu.multiples_plots(vu.one_train_plot, subject, data, measures, out_directory)


            

                



