import os
import pandas as pd
import numpy as np
from torch import load, device
from wandb_utils import load_df_from_wandb
from visu_utils import ROI_map, voxels_map

# project = "gaimee/neuroencoding_audio"
# outpath = '.' #'/home/maellef/projects/def-pbellec/maellef/projects/cNeuromod_encoding_2020/'
# df_save = os.path.join(outpath, 'all')

# runs_df = load_df_from_wandb(project)
# runs_df.to_csv(df_save, sep=';')
runs_df = pd.read_csv('./all', sep=';')

result_path = '/home/maelle/Results'
#runs_df = pd.read_csv('./allruns', sep=';')

# #best_roi
# df = roi_df
# bs = 70
# lr = 1e-4
# ks = 5
# wd = 1e-3
# patience = 15
# delta = 0
# wd_bool = df['wd'] == wd
# delta_bool = df['delta'] == delta
# patience_bool = df['patience'] == patience

for sub, (lr, ks, bs) in zip([4, 6], [(1e-05, 6, 80),(1e-05, 6, 70)]):
    sub_df = runs_df[runs_df['sub']==sub]
    bs_bool = sub_df['bs'] == bs
    lr_bool = sub_df['lr'] == lr
    ks_bool = sub_df['ks'] == ks
    roi_df = sub_df[(sub_df['scale']=='MIST_ROI')& bs_bool & lr_bool & ks_bool]
    vox_df = sub_df[(sub_df['scale']=='auditory_Voxels') & bs_bool & lr_bool & ks_bool]
    print(runs_df.shape, sub_df.shape, roi_df.shape, vox_df.shape)
    for i, df in enumerate([roi_df, vox_df]):
        layer_r2 = {'none':[], 'conv7':[], 'conv6':[], 'conv5':[], 'conv4':[]}
        for path, dirs, files in os.walk(result_path):
            for f in files:
                if f.find('_wbid') > -1:
                    start_id = f.find('_wbid')+len('_wbid')
                    end_id = f.find('_2022')
                    wandb_id = f[start_id:end_id]
                    if wandb_id in list(df['id']):
                        filepath = os.path.join(path, f)
                        data = load(filepath, map_location=device('cpu'))
                        layer_serie = df['finetuneStart'][df['id'] == wandb_id]
                        if pd.isnull(layer_serie.item()):
                            layer_r2['none'].append(data['test_r2'])
                        else : 
                            layer_r2[layer_serie.item()].append(data['test_r2'])                
        for layer, data in layer_r2.items():
            arr= np.array(data).reshape(len(data), -1)
            moy_arr = np.mean(arr, axis=0)
            if i == 0:
                rtitle = 'sub0{}_finetune_{}_roi_r2_map'.format(sub, layer)
                ROI_map(moy_arr, rtitle, result_path, threshold=0.05)
            elif i == 1:
                vtitle = 'sub0{}_finetune_{}_auditory_voxels_r2_map'.format(sub, layer)
                voxels_map(moy_arr, vtitle, result_path, threshold=0.05)





#-------analysis no training------------------------------------------------------------------------------------------------------
# noTraining_df = sub3_df[sub3_df['noTraining']==True]

# Training_df = sub3_df[(sub3_df['noTraining']==False) & (sub3_df['lr']==1e-4)]
# print(Training_df.shape)
# noInit_df = noTraining_df[noTraining_df['noInit']==True]
# noTrain_df = noTraining_df[noTraining_df['noInit']==False]

# for name, df in zip(['baseline', 'without training', 'without training + init of SoundNet'], [Training_df, noTrain_df, noInit_df]):
#     print(name)
#     mean_df = df.mean()
    
#     voxels_r2 = []
#     ROI_r2 = []
#     for path, dirs, files in os.walk(result_path):
#         for f in files:
#             if f.find('_wbid') > -1:
#                 end_id = f.find('_2022')
#                 wandb_id = f[len('_wbid'):end_id]
#                 if wandb_id in list(df['id']):
#                     filepath = os.path.join(path, f)
#                     data = load(filepath, map_location=device('cpu'))
#                     scale = df['scale'][df['id'] == wandb_id]
#                     if scale.array[0] == 'MIST_ROI' : 
#                         ROI_r2.append(data['test_r2'])
#                     else : 
#                         voxels_r2.append(data['test_r2'])

#     print(len(ROI_r2))
#     arr_vox = np.array(voxels_r2).reshape(len(voxels_r2), -1)
#     arr_roi = np.array(ROI_r2).reshape(len(ROI_r2), -1)
#     print(arr_roi.shape, arr_vox.shape)
#     vmoy_arr = np.mean(arr_vox, axis=0)
#     rmoy_arr = np.mean(arr_roi, axis=0)
#     print(rmoy_arr.shape, vmoy_arr.shape)

#     rtitle = 's03_{}_roi_r2_map'.format(name)
#     ROI_map(rmoy_arr, rtitle, result_path, threshold=0.05)
#     vtitle = 's03_{}_auditory_voxels_r2_map'.format(name)
#     voxels_map(vmoy_arr, vtitle, result_path, threshold=0.05)