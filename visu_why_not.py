import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch import load, device

from nilearn import image, plotting, datasets, surface
from nilearn.plotting import plot_stat_map, view_img, view_img_on_surf
from nilearn.regions import signals_to_img_labels
from nilearn.image import load_img, mean_img
from nilearn.input_data import NiftiMasker

from matplotlib import pyplot as plt 

path_auditory = '/home/maelle/Results/20210624early_stopping_auditory_Voxels_lr001_200epochs'
path_ROI = '/home/maelle/Results/20210624early_stopping_MIST_ROI_lr001_200epochs'

results_dir = [path_auditory, path_ROI]
tests = ['default', 'lrSch', 'trainPass']

metric = 'val_loss' #'trs'

for result_dir in results_dir:
    for sub in os.listdir(result_dir):
        sub_path = os.path.join(result_dir, sub)

        for file_name in os.listdir(sub_path):
            file_path = os.path.join(sub_path, file_name)
            if not os.path.isdir(file_path):
                if file_name.find('lrSch') > -1 : 
                    new_path = os.path.join(sub_path, 'lrSch', file_name)
                    os.rename(file_path, new_path)
                elif file_name.find('trainPass') > -1 : 
                    new_path = os.path.join(sub_path, 'trainPass', file_name)
                    os.rename(file_path, new_path)
                else : 
                    new_path = os.path.join(sub_path, 'default', file_name)
                    os.rename(file_path, new_path)

for result_dir in results_dir:
    for sub in os.listdir(result_dir):
        sub_path = os.path.join(result_dir, sub)
        f = plt.figure(figsize=(5, 15))

        for i, test in enumerate(tests) : 
            test_path = os.path.join(sub_path, test)
            plt.subplot(len(tests), 1, i+1)

            for file_name in os.listdir(test_path):
                if file_name[-3:] == '.pt':
                    file_path = os.path.join(test_path, file_name)
                    data = load(file_path)
                    x = np.array(data[metric])
                    plt.plot(x)


            plt.legend(test)
            #plt.subplots_adjust(left=0.05, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        
        fig_path = os.path.join(sub_path, sub+'_'+test)
        plt.savefig(fig_path)

                

        
        
