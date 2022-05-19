import os
import torch
import files_utils as fu

models_path = '/home/maelle/Results/best_models'

#load the model
for path, dirs, files in os.walk(models_path):
    for model in files:
        if '.pt' in model:
            model_path = os.path.join(path, model)
            model_data = torch.load(model_path, map_location=torch.device('cpu'))
            model_net = model_data['model']
            model_weights = model_data['net']
            model_net.load_state_dict(model_weights)

#load one run at a time
dataset_path = '/home/maelle/DataBase/fMRI_Embeddings'
stimuli_path = 

scales = ['auditory_Voxels', 'MIST_ROI']
subjects = ['sub-02']
datasets = ['friends']

runs_list = []
for scale in scales:
    for dataset in datasets:
        for sub in subjects:
            all_runs_path = os.path.join(dataset_path, scale, dataset, sub)
            fu.associate_stimuli_with_Parcellation(stimuli_path=)



#create a dataset for eval made of one run




