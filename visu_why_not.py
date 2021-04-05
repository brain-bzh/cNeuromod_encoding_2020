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

path_auditory = '/home/maelle/Results/20210404_generalisation_btw_friends_s1_s2_auditory_Voxels556_lr0.01_200epochs_embed2021norm'
path_ROI = '/home/maelle/Results/20210404_generalisation_btw_friends_s1_s2_MIST_ROI210_lr0.01_200epochs_embed2021norm'

results_dir = [path_auditory, path_ROI]

for result_dir in results_dir:
    for sub in os.listdir(result_dir):
        sub_path = os.path.join(result_dir, sub)

        sub_results = []
        for result_file in os.listdir(sub_path):
            result_path = os.path.join(sub_path, result_file)
            sub_results.append(load(result_path, map_location=device('cpu')))
        
        for result in sub_results : 
            print(result.keys())
