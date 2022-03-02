import os
import numpy as np
import pandas as pd
from torch import load, device
from nilearn import datasets, surface, plotting, regions, image, input_data
from matplotlib import pyplot as plt 

directory = '.'
voxel_fnull_path = os.path.join(directory, 'friends_auditory_Voxels_SoundNetEncoding_conv_0700520_1e-01_1e-04_1e-02_opt110_20220208-054633.pt')
voxel_fconv4_path = os.path.join(directory, 'friends_auditory_Voxels_SoundNetEncoding_conv_0700515_1e-01_1e-04_1e-02_opt110_f_conv4_20220208-143414.pt')
voxel_fconv5_path = os.path.join(directory, 'friends_auditory_Voxels_SoundNetEncoding_conv_0700515_1e-01_1e-04_1e-02_opt110_f_conv5_20220208-060110.pt')
MistROI_fnull_path = os.path.join(directory, 'friends_MIST_ROI_SoundNetEncoding_conv_0700515_5e-01_1e-04_1e-03_opt110_20220208-070809.pt')
MistROI_fconv4_path = os.path.join(directory, 'friends_MIST_ROI_SoundNetEncoding_conv_0700515_5e-01_1e-04_1e-03_opt110_f_conv4_20220208-114722.pt') 
MistROI_fconv5_path = os.path.join(directory, 'friends_MIST_ROI_SoundNetEncoding_conv_0700515_5e-01_1e-04_1e-03_opt110_f_conv5_20220208-130744.pt')
mistroifile = os.path.join(directory,"MIST_ROI.nii.gz")
mask = os.path.join(directory,'STG_middle.nii.gz')
tr = 1.49

def brain_3D_map(stat_img, title='', hemisphere='right', threshold=1.0, figure=None, output_file=None):
    fsaverage = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)
    plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi=hemisphere,title=title, colorbar=True,threshold=threshold, bg_map=fsaverage.sulc_right,output_file=output_file, figure=figure)

def voxels_map(data, title, out_directory, threshold):
    mymasker = input_data.NiftiMasker(mask_img=mask,standardize=False,detrend=False,t_r=tr,smoothing_fwhm=8)
    mymasker.fit()
    r2_stat = mymasker.inverse_transform(data)
    brain_3D_map(r2_stat, title=title, hemisphere='right', threshold=0.05, output_file=os.path.join(out_directory, title+'.png'))
    f = plt.figure()
    plotting.plot_stat_map(r2_stat, threshold = threshold, title=title, figure=f, colorbar=True)
    f.savefig(os.path.join(out_directory, title+'.png'))
    plt.close()

def ROI_map(data, title, out_directory, threshold, display_mode='z'):
    r2_img = regions.signals_to_img_labels(data,mistroifile)
    #r2_img.to_filename(os.path.join(out_directory, title))
    #data_map = image.load_img(os.path.join(out_directory, '{}.nii'.format(title)))
    #r2_img = data_map
    f = plt.figure()
    plotting.plot_stat_map(r2_img,display_mode=display_mode ,cut_coords=6,figure=f, threshold=threshold, colorbar=True)
    f.savefig(os.path.join(out_directory, title))
    plt.close()

filepath = voxel_fconv4_path
data = load(filepath, map_location=device('cpu'))

model = data['model']
net = data['net']  #net.state_dict()
last_epoch = data['epoch']
train_loss = data['train_loss']
train_r2_max = data['train_r2_max']
train_r2_mean = data['train_r2_mean'] 
val_loss = data['val_loss'] 
val_r2_max = data['val_r2_max']
val_r2_mean = data['val_r2_mean']
test_loss = data['test_loss']
test_r2 = data['test_r2']
test_r2_max = data['test_r2_max']
test_r2_mean = data['test_r2_mean']
lrs = data['lrs']
training_time = data['training_time']
hyperparameters = data['hyperparameters']
data_processing = data['data_processing'] 
data_selection = data['data_selection']

test_r2 = test_r2.reshape(1,-1)
#ROI_map(data=test_r2, title="MISTROI_R2_map_finetuned_conv4", out_directory=directory, threshold=0.05)
voxels_map(data=test_r2, title="voxel_R2_map_finetuned_conv4", out_directory=directory, threshold=0)