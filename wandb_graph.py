import pandas as pd 
import os
import wandb
from matplotlib import pyplot as plt

api = wandb.Api()
selected_scale = 'auditory_Voxels' #'MIST_ROI' 
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

#--------------------------------------------------------------------------------
r2_threshold = 300
sorted_df = analysis_df.sort_values(by=['val r2 max'], ascending=False)

bs_score = {1:[],10:[],30:[],70:[]}
ks_score = {1:[],3:[],5:[],9:[]}
lr_score = {1e-2:[],1e-3:[],1e-4:[]}
wd_score = {1e-2:[],1e-3:[],1e-4:[]}
patience_es_score = {10:[],15:[],20:[]}
delta_es_score = {0:[], 1e-1:[], 5e-1:[]}

for n in range(5, r2_threshold+1, 5):

    hyperparameters = {
    'bs' : {1:0,10:0,30:0,70:0},
    'ks' : {1:0,3:0,5:0,9:0},
    'lr' : {1e-2:0,1e-3:0,1e-4:0},
    'wd' : {1e-2:0,1e-3:0,1e-4:0},
    'patience' : {10:0,15:0,20:0},
    'delta' : {0:0, 1e-1:0, 5e-1:0}}

    best_runs = sorted_df.head(n=n)
    for idx, row in best_runs.iterrows() : 
        for key, value in hyperparameters.items():
            param = row[key]
            hyperparameters[key][param] += 1

    for score, hp in zip([bs_score, ks_score, lr_score, wd_score, patience_es_score, delta_es_score], hyperparameters.keys()):
        hp_dict = hyperparameters[hp]
        for key, value in hp_dict.items():
            score[key].append(value*100/n)

scores = {
'batchsize_frequency' : bs_score,
'kernelsize_frequency': ks_score,
'learning_rate': lr_score,
'weight_decay': wd_score,
'patience_(early_stopping)': patience_es_score,
'delta_(early_stopping)': delta_es_score
}

r2_scores = {
'train r2 mean' : None,
'train r2 max' : None,
'val r2 mean' : None,
'val r2 max' : None,
'test r2 mean' : None,
'test r2 max' : None
}

for criterium, dico in r2_scores.items():
    r2_scores[criterium] = list(sorted_df[criterium])

def plot_dict_data(dico, name, out_directory, x_label='', y_label='', colors = ['b', 'c', 'm', 'r', 'g', 'k']) : 
    f = plt.figure()
    legends = []
    legend_values = list(dico.keys())
    data = list(dico.values())
    for color, legend, y in zip(colors, legend_values, data):
        x = range(len(y))
        plt.plot(x[:500], y[:500], color)
        legends.append('value : {}'.format(legend))
    plt.legend(legends, loc='lower left')
    plt.xlabel('number of runs')
    plt.ylabel('frequency amongst the runs')
    plt.title(name)
    f.savefig(os.path.join(out_directory, name+'.jpg'))
    plt.close()

if __name__ == "__main__":
    outpath = os.path.join('/home/maelle/Results/20220126_Hypertraining_analysis', selected_scale)

    plot_dict_data(r2_scores, '{}_r2_relative_to_500_runs_ordered_by_val_r2_max'.format(selected_scale, criterium), outpath)

    # for hp_name, score in scores.items():
    #     plot_dict_data(score, '{}_{}_relative_to_number_of_selected_best_runs'.format(selected_scale, hp_name), outpath)

#runs_df.to_csv("project.csv")
#bs = 70, ks = 5, lr = 0.0001, wd = 0.01, patience = 15, delta = 0.1




 