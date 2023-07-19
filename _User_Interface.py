#test test test Coucou

#Generic
import os
import warnings
import numpy as np
from datetime import datetime
#Get input & preprocess
import files_utils as fu
import visualisation_utils as vu
from model_training import model_training
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

data_selection = {
    'films':[life, hidden],
    'subjects':[0,1,2,3,4],
    'sessions':[1,2]
}

data_processing = {
    'scale': ['MIST_ROI', 'auditory_Voxels'], #'MIST_ROI'
    'tr' : 1.49,
    'sr' : 22050,
    'selected_ROI': [None, None],#[141,152,153,169,170,204,205]
    'nroi': [210, 556] #210 #7                            
}

#model parameters
allfmrihidden=[1000]
models = [encod.SoundNetEncoding_conv]
kernel_sizes=[5]

training_hyperparameters = {
    'batchsize':30,
    'train_percent':1.0,
    'test_percent':0.5,
    'val_percent':0.5,
    #to change, good for now
    'diff_sess_for_train_test':True,

    'gpu':False,
    'lr':0.01,
    'nbepoch': 100,
    'mseloss':nn.MSELoss(reduction='sum'),
    'weight_decay':[1e-2],
    'decoupled_weightDecay' : [True],
    'power_transform' : [False]
    #'warm_restart' : [True, False] à tester
    #'early_stopping':EarlyStopping(patience=10, verbose=True,delta=1e-6)
    #problem, that stop the training of any following test
    #to look how to implement a more interactable early_stopping
}

#----------------------Path?_loop-----------------------------------------------------------------------
if __name__ == "__main__":
    for scale, select_roi, nroi in zip(data_processing['scale'],  data_processing['selected_ROI'], data_processing['nroi']):
        #paths
        outpath = "/home/maelle/Results/"
        stimuli_path = '/home/maelle/Database/fMRI_datasets/cneuromod/movie10/stimuli' #'/home/brain/Data_Base/cneuromod/movie10/stimuli' 
        path_embed = '/home/maelle/Database/Embeddings/cNeuromod/embed_2020_norm'
        embed_used = os.path.basename(path_embed).replace('_', '')
        path_parcellation = os.path.join(path_embed, scale) #'/home/brain/Data_Base/movie10_parc'
        all_subs_files = fu.associate_stimuli_with_Parcellation(stimuli_path, path_parcellation)
        resultpath = outpath+dt_string+"_generalisation_btw_sessions_{}{}_lr0.01_{}epochs_{}".format(scale,nroi, training_hyperparameters['nbepoch'],embed_used)
        fu.create_dir_if_needed(outpath)

#--------------------------TRAINING LOOP-------------------------------------------------------------------
        for subject in data_selection['subjects']:
            outpath_sub = os.path.join(resultpath, 'subject_'+str(subject))
            fu.create_dir_if_needed(outpath_sub)

            for film in data_selection['films']:
                outpath_film = os.path.join(outpath_sub, film)
                fu.create_dir_if_needed(outpath_film)

                for model in models:
                    for fmrihidden in allfmrihidden:
                        for kernel_size in kernel_sizes:
                            for weight_decay in training_hyperparameters['weight_decay']:
                                for decoupled_weightDecay in training_hyperparameters['decoupled_weightDecay']:
                                    for power_transform in training_hyperparameters['power_transform']:
                                        dp = data_processing.copy()
                                        dp['scale'] = scale
                                        dp['selected_ROI'] = select_roi
                                        dp['nroi'] = nroi  

                                        ds = data_selection.copy()
                                        ds['all_data'] = all_subs_files
                                        ds['subject'] = subject
                                        ds['film'] = film

                                        train_HP = training_hyperparameters.copy()
                                        train_HP['model'] = model
                                        train_HP['fmrihidden'] = fmrihidden
                                        train_HP['kernel_size'] = kernel_size
                                        train_HP['weight_decay'] = weight_decay
                                        train_HP['decoupled_weightDecay'] = decoupled_weightDecay
                                        train_HP['power_transform'] = power_transform

                                        model_training(outpath_film, ds, dp, train_HP)

        fu.rename_object(resultpath, 'subject_', fu.cNeuromod_subject_convention, objects=['dirs'])
        
        #results visu

        # out_directory = os.path.join(resultpath, 'analysis')
        # fu.create_dir_if_needed(out_directory)
        # auditorymask='STG_middle.nii.gz'

        # all_data = vu.construct_data_dico(data_path=resultpath, extension='.pt', criterion='sub', target='ks')
        # #all_maps = construct_data_dico('film', '.gz', target_path)
        # all_data = vu.sort_by_target_value(all_data)
        # for sub, films in all_data.items():
        #     all_loaded = [(dir_name, load(file_path, map_location=device('cpu')), target_data) for (dir_name, file_path, target_data) in films]
        #     all_data[sub] = all_loaded

        # all_data = vu.subdivise_dict(all_data)

        # measures = ["loss", "r2_max", "r2_mean"]
        # for subject, data in all_data.items():
        #     vu.multiples_plots(vu.one_train_plot, subject, data, measures, out_directory)


            

                



