import pandas as pd 
import os
import wandb
from matplotlib import pyplot as plt

#derived from wandb documentation
def load_df_from_wandb(project):
    api = wandb.Api()
    runs = api.runs(project)
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
    return(runs_df)

def order_results_files_from_wandb(results_path, wandb_path, subject, scale, project=None, csv=None) : 
    if project == None : 
        runs_df = pd.read_csv(csv, sep=';')
    elif csv = None :
        runs_df = load_df_from_wandb(project)
    
    


if __name__ == "__main__":
    # Project is specified by <entity/project-name>
    project = "gaimee/neuroencoding_audio"
    outpath = '/home/maellef/projects/def-pbellec/maellef/projects/cNeuromod_encoding_2020/'
    df_save = os.path.join(outpath, 'configs_from_HPtrain_2021')

    runs_df = load_df_from_wandb(project)
    runs_df.to_csv(df_save, sep=';')