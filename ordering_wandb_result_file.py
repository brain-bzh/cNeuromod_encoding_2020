import pandas as pd 
import os
import wandb
from torch import load, device
import numpy as np

api = wandb.Api()
selected_scale = 'MIST_ROI' #'auditory_Voxels' 
# Project is specified by <entity/project-name>
runs = api.runs("gaimee/neuroencoding_audio")

summary_list, config_list, name_list, id_list = [], [], [], []

for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)
    id_list.append(run.id)

summary_pd = pd.DataFrame(summary_list)
config_pd = pd.DataFrame(config_list)
name_pd = pd.Series(name_list, name='name')
id_pd = pd.Series(id_list, name='id')
runs_df = pd.concat((id_pd, name_pd, config_pd, summary_pd), axis=1)

analysis_df = runs_df[runs_df['scale'] == selected_scale]
#analysis_df = df[df['finetuneStart'] == None]
sorted_df = analysis_df.sort_values(by=['val r2 max'], ascending=False)

# pour un id, récupère la série de "test_r2"

wandb_path = '/home/maellef/scratch/wandb/wandb'
results_path = '/home/maellef/scratch/Results'
subject = 'sub-03'

ordered_runs = []
for _, row in sorted_df.iterrows():
    outfile_name = '{:03}{:02}{:02}'.format(row['bs'], row['ks'], row['patience'])
    outfile_name +='{:.0e}'.format(row['delta'])[-3:]+'{:.0e}'.format(row['lr'])[-3:]+'{:.0e}'.format(row['wd'])[-3:]+'_opt'

    outfile_name = outfile_name+'1' if row['decoupledWD'] else outfile_name+'0'
    outfile_name = outfile_name+'1' if row['lrScheduler'] else outfile_name+'0'
    outfile_name = outfile_name+'1' if row['powerTransform'] else outfile_name+'0'
    outfile_name = outfile_name+'_f_'+ row['finetuneStart'] if row['finetuneStart'] != None else outfile_name
    delta = '{:.0e}'.format(row['delta'])
    wandb_id = row['id']
    #print(wandb_id, row['name'], row['finetuneStart'], type(row['finetuneStart']))
    if row['finetuneStart'] != None : 
        pass
    else : 
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
        try:
            if delta[0] == '5':
                selected_file = os.path.join(id_directory, results_files[1])
            else :         
                selected_file = os.path.join(id_directory, results_files[0])
            ordered_runs.append(selected_file)
        
        except IndexError : 
            pass

all_data = np.array([]).reshape(0,210)
for datafile in ordered_runs : 
    data = load(datafile, map_location=device('cpu'))
    data = data['test_r2'].reshape(1,-1)
    all_data = np.concatenate((all_data, data), axis=0)

outpath = '/home/maellef/projects/def-pbellec/maellef/projects/cNeuromod_encoding_2020/'
array_save = os.path.join(outpath, 'ordered_data_from_HPtrain_2021')
df_save = os.path.join(outpath, 'ordered_configs_from_HPtrain_2021')
print(all_data.shape)
np.save(array_save, all_data)
analysis_df.to_csv(df_save, sep=';')


