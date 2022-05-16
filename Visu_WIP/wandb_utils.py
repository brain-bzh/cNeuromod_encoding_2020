import pandas as pd 
import numpy as np
import os
import wandb
from matplotlib import pyplot as plt
from visu_utils import parameter_mode_in_dataset
from visu_utils import ROI_map
from torch import load, device

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
    elif csv == None :
        runs_df = load_df_from_wandb(project)

if __name__ == "__main__":
    # Project is specified by <entity/project-name>
    # project = "gaimee/neuroencoding_audio"
    # outpath = '.' #'/home/maellef/projects/def-pbellec/maellef/projects/cNeuromod_encoding_2020/'
    # df_save = os.path.join(outpath, 'allruns')

    #runs_df = load_df_from_wandb(project)
    #runs_df.to_csv(df_save, sep=';')
    runs_df = pd.read_csv('./subs_2346', sep=';')
    
    subs = [2] #4, 6]
    scales = ['MIST_ROI', 'auditory_Voxels']
    lrs = [1e-4, 1e-5, 1e-6]
    kss = [5,6,7]
    bss = [60,70,80]
    configs = []
    for lr in lrs:
        for ks in kss:
            for bs in bss:
                configs.append((lr, ks, bs))

    for sub, (best_lr, best_ks, best_bs) in zip(subs, [(1e-05,7,80),(1e-05, 6, 80), (1e-05, 6, 70)]):
    #for sub in subs : 
        for scale in scales:
            print(sub, scale)
            sub_df = runs_df[runs_df['sub'] == sub]
            analysis_df = sub_df[sub_df['scale'] == scale]
            sorted_df_1 = analysis_df.sort_values(by=['val r2 max'], ascending=False)
            for hp in ['lr', 'ks', 'bs']:
                parameter_mode_in_dataset(sorted_df_1.head(20), hp, 'individual runs')


            moy = []
            for i, (lr, ks, bs) in enumerate(configs):
                #print('config {} - lr : {}, ks : {}, bs : {}'.format(i, lr, ks,bs))
                temp_df = analysis_df[(analysis_df['lr'] == lr) & (analysis_df['ks'] == ks) & (analysis_df['bs'] == bs)]
                moy.append(temp_df.mean())
            mean_df = pd.DataFrame(moy)
            
            sorted_df_2 = mean_df.sort_values(by=['val r2 max'], ascending=False)
            for hp in ['lr', 'ks', 'bs']:
                parameter_mode_in_dataset(sorted_df_1.head(10), hp, 'grouped runs (mean) by configs')
            
            best_configs = analysis_df[(analysis_df['lr'] == best_lr) & (analysis_df['ks'] == best_ks) & (analysis_df['bs'] == best_bs)]
            best_mean = best_configs.mean()
            best_max = best_configs.max()
            best_min = best_configs.min()

            print('max validation r² : {} (+{}-{})'.format(best_mean['val r2 max'], best_max['val r2 max'], best_min['val r2 max']))
            print('max test r² : {} (+{}-{})'.format(best_mean['test r2 max'], best_max['test r2 max'], best_min['test r2 max']))

            print('mean validation r² : {} (+{}-{})'.format(best_mean['val r2 mean'], best_max['val r2 mean'], best_min['val r2 mean']))
            print('mean test r² : {} (+{}-{})'.format(best_mean['test r2 mean'], best_max['test r2 mean'], best_min['test r2 mean']))
            
            result_path = '/home/maelle/Results'
            ROI_r2 = []
            for path, dirs, files in os.walk(result_path):
                for f in files:
                    if f.find('_wbid') > -1:
                        end_id = f.find('_2022')
                        wandb_id = f[len('_wbid'):end_id]
                        if wandb_id in list(best_configs['id']):
                            filepath = os.path.join(path, f)
                            data = load(filepath, map_location=device('cpu'))
                            ROI_r2.append(data['test_r2'])
            print(len(ROI_r2))
            arr_roi = np.array(ROI_r2).reshape(len(ROI_r2), -1)
            print(arr_roi.shape)
            rmoy_arr = np.mean(arr_roi, axis=0)
            print(rmoy_arr.shape)
            rtitle = '{}_roi_r2_map'.format(sub)
            ROI_map(rmoy_arr, rtitle, result_path, threshold=0.05)



    #[Unnamed: 0, id, name, bs, hs, ks, lr, sr, tr, wd, gpu, sub, comet, delta, scale, wandb, noInit, select, val100, dataset, 
    # nbepoch, test100, evalData, patience, train100, trainData, noTraining, decoupledWD, lrScheduler, outputLayer, sessionsEval, 
    # finetuneDelay, finetuneStart, sessionsTrain, powerTransform, epochStart, train loss, test r2 max, test r2 mean, _step, 
    # learning rate, nb epochs, test loss, _timestamp, val r2 mean, train r2 mean, _runtime, val loss, val r2 max, train r2 max]