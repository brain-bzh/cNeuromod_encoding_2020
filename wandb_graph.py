import pandas as pd 
import wandb
api = wandb.Api()

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
MIST_df = runs_df[runs_df['scale'] == 'MIST_ROI']
voxels_df = runs_df[runs_df['scale'] == 'auditory_Voxels']

#--------------------------------------------------------------------------------

analysis_df = voxels_df
best_runs_nb = 100

hyperparameters = {
'bs' : {1:0,10:0,30:0,70:0},
'ks' : {1:0,3:0,5:0,9:0},
'lr' : {1e-2:0,1e-3:0,1e-4:0},
'wd' : {1e-2:0,1e-3:0,1e-4:0},
'patience' : {10:0,15:0,20:0},
'delta' : {0:0, 1e-1:0, 5e-1:0}
}

sorted_df = analysis_df.sort_values(by=['val r2 max'], ascending=False)
best_runs = sorted_df.head(n=best_runs_nb)

for idx, row in best_runs.iterrows() : 
    for key, value in hyperparameters.items():
        param = row[key]
        hyperparameters[key][param] += 1

for key, value in hyperparameters.items():
    print (key, value)


#runs_df.to_csv("project.csv")
#bs = 70, ks = 5, lr = 0.0001, wd = 0.01, patience = 15, delta = 0.1




 