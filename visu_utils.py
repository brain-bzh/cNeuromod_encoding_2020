import os
#import numpy as np
#import pandas as pd
#import pickle
from torch import load, device
from nilearn import datasets, surface, plotting, regions, image, input_data
from matplotlib import pyplot as plt 
#from files_utils import create_dir_if_needed, print_dict, extract_value_from_string

mistroifile = "/home/maelle/DataBase/fMRI_parcellations/MIST_parcellation/Parcellations/MIST_ROI.nii.gz"
mask='STG_middle.nii.gz'
tr = 1.49
criteria = 'test_r2'

def brain_3D_map(stat_img, title='', hemisphere='right', threshold=1.0, figure=None, output_file=None):
    fsaverage = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)
    plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi=hemisphere,title=title, colorbar=True,threshold=threshold, bg_map=fsaverage.sulc_right,output_file=output_file, figure=figure)

def voxels_map(filename, title, out_directory, threshold):
    data = load(filename, map_location=device('cpu'))
    mymasker = input_data.NiftiMasker(mask_img=mask,standardize=False,detrend=False,t_r=tr,smoothing_fwhm=8)
    mymasker.fit()
    r2_stat = mymasker.inverse_transform(data[criteria].reshape(1,-1))
    brain_3D_map(r2_stat, title=title, hemisphere='right', threshold=0.05, output_file=os.path.join(out_directory, title+'.png'))
    f = plt.figure()
    plotting.plot_stat_map(r2_stat, threshold = threshold, title=title, figure=f)
    f.savefig(os.path.join(out_directory, title+'.png'))
    plt.close()

def ROI_map(filename, title, out_directory, threshold, display_mode='z'):
    data = load(filename, map_location=device('cpu'))
    r2_img = regions.signals_to_img_labels(data[criteria].reshape(1,-1),mistroifile)
    #r2_img.to_filename(os.path.join(out_directory, title))
    #data_map = image.load_img(os.path.join(out_directory, '{}.nii'.format(title)))
    #r2_img = data_map
    f = plt.figure()
    plotting.plot_stat_map(r2_img,display_mode=display_mode ,cut_coords=6,figure=f, threshold=threshold)
    f.savefig(os.path.join(out_directory, title))
    plt.close()

if __name__ == "__main__":
    voxels = [
    "/home/maelle/Results/20220126_Hypertraining_analysis/auditory_Voxels/balmy cosmos/friends_auditory_Voxels_SoundNetEncoding_conv_0700510+00-03-04_opt110_20211120-00:20:00.pt",
    "/home/maelle/Results/20220126_Hypertraining_analysis/auditory_Voxels/easy vortex/friends_auditory_Voxels_SoundNetEncoding_conv_0700515-01-04-04_opt110_20211123-04:13:39.pt",
    "/home/maelle/Results/20220126_Hypertraining_analysis/auditory_Voxels/twilight paper/friends_auditory_Voxels_SoundNetEncoding_conv_0700915-01-03-02_opt110_20211116-03:37:32.pt",
    "/home/maelle/Results/20220126_Hypertraining_analysis/auditory_Voxels/twilight paper/friends_auditory_Voxels_SoundNetEncoding_conv_0700915-01-03-02_opt110_20211116-05:42:37.pt"
    ]

    MIST_ROI_files = [
    "/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/vibrant meadow/friends_MIST_ROI_SoundNetEncoding_conv_0700915+00-04-02_opt110_20211130-20:31:24.pt",
    "/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/effortless night/friends_MIST_ROI_SoundNetEncoding_conv_0700910-01-04-02_opt110_20211130-18:37:53.pt",
    "/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/denim rain/friends_MIST_ROI_SoundNetEncoding_conv_0700515-01-04-03_opt110_20211201-18:34:53.pt",
    "/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/denim rain/friends_MIST_ROI_SoundNetEncoding_conv_0700515-01-04-03_opt110_20211201-21:07:28.pt",
    "/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/celestial snow/friends_MIST_ROI_SoundNetEncoding_conv_0700910-01-04-03_opt110_20211201-13:20:33.pt",
    '/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/celestial snow/friends_MIST_ROI_SoundNetEncoding_conv_0700910-01-04-03_opt110_20211201-15:13:50.pt'
    ]  

for j, scale_filesList in enumerate([voxels, MIST_ROI_files]):
    for i, filename in enumerate(scale_filesList) : 
        path, pt_file = os.path.split(filename)
        name, _ = os.path.splitext(pt_file)
        out_directory, prefix = os.path.split(path)
        title = '{}_{}_{}_r2_map'.format(prefix, i, name)

        if j==0 : 
            voxels_map(filename, title, out_directory, threshold=0.02)
        else :
            ROI_map(filename, title, out_directory, threshold=0)