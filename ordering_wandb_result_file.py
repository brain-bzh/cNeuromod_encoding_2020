import pandas as pd 
import os
from torch import load, device
import numpy as np
#from wandb_utils import load_df_from_wandb #desactivate if you're working on ComputeCanada


selected_scale = 'MIST_ROI' #'auditory_Voxels' 
#runs_df = load_df_from_wandb("gaimee/neuroencoding_audio")

runs_df = pd.read_csv('/home/maellef/projects/def-pbellec/maellef/projects/cNeuromod_encoding_2020/configs_from_HPtrain_2021', sep=';')
selected_df = runs_df[runs_df['scale'] == selected_scale]
analysis_df = selected_df[selected_df['finetuneStart'].isnull()]
sorted_df = analysis_df.sort_values(by=['val r2 max'], ascending=False)

# pour un id, récupère la série de "test_r2"

wandb_path = '/home/maellef/scratch/wandb/wandb'
results_path = '/home/maellef/scratch/Results'
subject = 'sub-03'

indexes = []
ordered_runs = []
for i, (idx, row) in enumerate(sorted_df.iterrows()):
    outfile_name = '{:03}{:02}{:02}'.format(row['bs'], row['ks'], row['patience'])
    outfile_name +='{:.0e}'.format(row['delta'])[-3:]+'{:.0e}'.format(row['lr'])[-3:]+'{:.0e}'.format(row['wd'])[-3:]+'_opt'

    outfile_name = outfile_name+'1' if row['decoupledWD'] else outfile_name+'0'
    outfile_name = outfile_name+'1' if row['lrScheduler'] else outfile_name+'0'
    outfile_name = outfile_name+'1' if row['powerTransform'] else outfile_name+'0'
    outfile_name = outfile_name+'_f_'+ row['finetuneStart'] if not np.isnan(row['finetuneStart']) else outfile_name
    delta = '{:.0e}'.format(row['delta'])
    wandb_id = row['id']

    results_files = []
    for filename in os.listdir(wandb_path):
        if filename.find(wandb_id) > -1 : 
            completion_day = filename[12:20]
            completion_hour = filename[21:27]
            for directories in os.listdir(results_path):
                if directories.find(completion_day) >-1 :     
                    id_directory = os.path.join(results_path, directories, subject)
                    for result_file in os.listdir(id_directory):
                        if result_file.find(outfile_name)>-1:
                            results_files.append(result_file)
    
    print(results_files)
    try:
        if delta[0] == '5':
            selected_file = os.path.join(id_directory, results_files[1])
        else :         
            selected_file = os.path.join(id_directory, results_files[0])
        ordered_runs.append(selected_file)
        indexes.append([i, idx])
    
    except IndexError : 
        pass

all_data = np.array([]).reshape(0,210)
for datafile in ordered_runs : 
    data = load(datafile, map_location=device('cpu'))
    data = data['test_r2'].reshape(1,-1)
    all_data = np.concatenate((all_data, data), axis=0)

indexes = np.array(indexes).reshape(-1,2)
print(indexes.shape, all_data.shape)
print(indexes)
all_data = np.concatenate((indexes, all_data), axis=1)
print(all_data[:10])

outpath = '/home/maellef/projects/def-pbellec/maellef/projects/cNeuromod_encoding_2020/'
array_save = os.path.join(outpath, 'ordered_data_from_HPtrain_2021')
df_save = os.path.join(outpath, 'ordered_configs_from_HPtrain_2021')
np.save(array_save, all_data)
analysis_df.to_csv(df_save, sep=';')


