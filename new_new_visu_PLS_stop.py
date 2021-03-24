import os
import numpy as np
import pandas as pd
import pickle
from torch import load, device

from files_utils import create_dir_if_needed, print_dict, extract_value_from_string

from nilearn import image, plotting, datasets, surface
from nilearn.plotting import plot_stat_map, view_img, view_img_on_surf
from nilearn.regions import signals_to_img_labels
from nilearn.image import load_img, mean_img
from nilearn.input_data import NiftiMasker

from matplotlib import pyplot as plt 

def one_train_plot_TEST(data, measure, colors = ['b', 'g', 'm', 'r']) : 
    legends = []
    for color, value in zip(colors, data):
        cdt = value['condition']

        train_data = value['train_'+str(measure)]#['mean']
        #(train_min, train_max) = value['train_'+str(measure)]['error_bar']

        val_data = value['val_'+str(measure)]#['mean']
        #(val_min, val_max) = value['val_'+str(measure)]['error_bar']

        plt.plot(train_data, color+'-')
        # plt.errorbar(range(len(train_data)), train_data, fmt=color+'-')
        plt.plot(val_data, color+'--')
        # plt.errorbar(range(len(val_data)), val_data, fmt=color+'--')
        
        legends.append(cdt+'_Train')
        legends.append(cdt+'_Val')
    plt.legend(legends, loc='upper left', bbox_to_anchor=(1,1))
  
  
datapath = '/home/maelle/Results/20210321_generalisation_btw_sessions_MIST_ROI210_lr0.01_100epochs_embed2020norm'
out_directory = '/home/maelle/Results/analysis_212103_Nicolas/gros_test/'
prefix = '2020_MIST_ROI210_lr10e-3_sub0_100epochs_GeneSess_'
create_dir_if_needed(out_directory)

measures = ["loss", "r2_max", "r2_mean"]
step_train = ["train", "val", "test"]

network_keys = []
for measure in measures:
    for step in step_train:
        network_keys.append(step+'_'+measure)

all_sub_data = {}
all_sub_maps = {}
for sub in os.listdir(datapath):
    sub_path = os.path.join(datapath, sub)
    subdata = []
    submaps = []
    for film in os.listdir(sub_path):
        film_path = os.path.join(sub_path, film)
        for data_file in os.listdir(film_path):
            file_path = os.path.join(film_path, data_file)
            basename, ext = os.path.splitext(data_file)
            if ext == '.pt':
                data = load(file_path, map_location=device('cpu'))
                data['condition'] = film
                subdata.append(data)
            elif ext == '.gz':
                data_map = image.load_img(file_path)
                submaps.append((film, data_map))
    all_sub_data[sub] = subdata
    all_sub_maps[sub] = submaps
    
#------------------plot train and val-------------------------------
f = plt.figure(figsize=(5*len(measures)+15, 7*len(all_sub_data)))

for i, (sub, films_data) in enumerate(all_sub_data.items()):
    for j, measure in enumerate(measures):
        ax = plt.subplot(len(all_sub_data),len(measures),len(measures)*i+(j+1))
        one_train_plot_TEST(films_data, measure)
        plt.title(str(measure)+
        ' for training (decoupled WD), for '+str(sub))

plt.subplots_adjust(left=0.05, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
f.savefig(os.path.join(out_directory, prefix+'ks_5'))
plt.close()

#------------------r2 ROI map-------------------------------
r2_tab = [(subject, r2_values) for subject, r2_values in all_sub_maps.items()]

f2 = plt.figure(figsize=(10,30))

for i, (sub, datafilms) in enumerate(r2_tab):
    for j, (film, data_map) in enumerate(datafilms):
        r2_img = data_map
        ax = plt.subplot(10,1, (2*i+j+1))
        title = 'R2 Map for {} in {}'.format(film, sub)
        plotting.plot_stat_map(r2_img,display_mode='z',cut_coords=5,figure=f2,axes=ax, title=title, threshold=0)

f2.savefig(os.path.join(out_directory, prefix+'_r2_maps'))
plt.close()

#-------------R2 voxels map--------------------------------------------
#         ks_map = [(wd, data) for ([ks, wd], data) in all_maps if ks==first_ks]
#         auditorymask='STG_middle.nii.gz'
#         f3 = plt.figure(figsize=(10,10))
#         for j, training_data in enumerate(ks_map):
#             ax = plt.subplot(4,1,j+1)
#             target = training_data[0]
#             data = training_data[1]
#             title = 'R2 Map in bourne : '+str(target)

#             mymasker = NiftiMasker(mask_img=auditorymask,standardize=False,detrend=False,t_r=1.49,smoothing_fwhm=8)
#             mymasker.fit()

#             r2_stat = mymasker.inverse_transform(data['test_r2'].reshape(1,-1))

#             #brain_3D_map(r2_stat, title=title, hemishpere='right', threshold=0.05, output_file=map_path+'.png')
#             plotting.plot_stat_map(r2_stat, threshold = 0.02, title=title, figure=f3, axes=ax)
#         f3.savefig(out_directory+'/'+prefix+'_stat_map_AudVox.png')
#         plt.close()