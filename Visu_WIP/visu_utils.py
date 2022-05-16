import os
import numpy as np
import statistics
#import pandas as pd
import pickle
from torch import load, device
from nilearn import datasets, surface, plotting, regions, image, input_data
from matplotlib import pyplot as plt

mistroifile = "/home/maelle/DataBase/fMRI_parcellations/MIST_parcellation/Parcellations/MIST_ROI.nii.gz"
mask='STG_middle.nii.gz'
tr = 1.49
criteria = 'test_r2'

#----------------------------------------------------------------------------------------------------------------------------
def brain_3D_map(stat_img, title='', hemisphere='right', threshold=1.0, figure=None, output_file=None):
    fsaverage = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)
    plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi=hemisphere,title=title, colorbar=True,threshold=threshold, bg_map=fsaverage.sulc_right,output_file=output_file, figure=figure)

def voxels_map(data, title, out_directory, threshold):
    #data = load(filename, map_location=device('cpu'))
    mymasker = input_data.NiftiMasker(mask_img=mask,standardize=False,detrend=False,t_r=tr,smoothing_fwhm=8)
    mymasker.fit()
    r2_stat = mymasker.inverse_transform(data.reshape(1,-1))
    r2_stat.to_filename(os.path.join(out_directory, title))
    brain_3D_map(r2_stat, title=title, hemisphere='right', threshold=0.05, output_file=os.path.join(out_directory, title+'.png'))
    # f = plt.figure()
    # plotting.plot_stat_map(r2_stat, threshold = threshold, title=title, figure=f, colorbar=True, vmax=0.42)
    # f.savefig(os.path.join(out_directory, title+'.png'))
    # plt.close()

def ROI_map(data, title, out_directory, threshold, display_mode='z'):
    #data = load(data, map_location=device('cpu'))
    #r2_img = regions.signals_to_img_labels(data[criteria].reshape(1,-1),mistroifile)
    r2_img = regions.signals_to_img_labels(data.reshape(1,-1),mistroifile)
    #r2_img.to_filename(os.path.join(out_directory, title))
    #data_map = image.load_img(os.path.join(out_directory, '{}.nii'.format(title)))
    #r2_img = data_map
    f = plt.figure()
    plotting.plot_stat_map(r2_img,display_mode=display_mode ,cut_coords=6,figure=f, threshold=threshold, colorbar=True, vmax=0.40)
    f.savefig(os.path.join(out_directory, title))
    plt.close()

def parameter_mode_in_dataset(dataframe, parameter, name = 'this subset'):
    a = dataframe[parameter].value_counts(normalize=True)
    print(a)
    try : 
        best_hp = statistics.mode(dataframe[parameter])
        print('dominant value of {} in {} : {}'.format(parameter, name, best_hp))
    except statistics.StatisticsError:
        print('no dominant value was found for {} in {}'.format(parameter, name))

if __name__ == "__main__":
    # voxels = [
    # "/home/maelle/Results/20220126_Hypertraining_analysis/auditory_Voxels/balmy cosmos/friends_auditory_Voxels_SoundNetEncoding_conv_0700510+00-03-04_opt110_20211120-00:20:00.pt",
    # "/home/maelle/Results/20220126_Hypertraining_analysis/auditory_Voxels/easy vortex/friends_auditory_Voxels_SoundNetEncoding_conv_0700515-01-04-04_opt110_20211123-04:13:39.pt",
    # "/home/maelle/Results/20220126_Hypertraining_analysis/auditory_Voxels/twilight paper/friends_auditory_Voxels_SoundNetEncoding_conv_0700915-01-03-02_opt110_20211116-03:37:32.pt",
    # "/home/maelle/Results/20220126_Hypertraining_analysis/auditory_Voxels/twilight paper/friends_auditory_Voxels_SoundNetEncoding_conv_0700915-01-03-02_opt110_20211116-05:42:37.pt"
    # ]
# 
    # MIST_ROI_files = [
    # "/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/vibrant meadow/friends_MIST_ROI_SoundNetEncoding_conv_0700915+00-04-02_opt110_20211130-20:31:24.pt",
    # "/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/effortless night/friends_MIST_ROI_SoundNetEncoding_conv_0700910-01-04-02_opt110_20211130-18:37:53.pt",
    # "/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/denim rain/friends_MIST_ROI_SoundNetEncoding_conv_0700515-01-04-03_opt110_20211201-18:34:53.pt",
    # "/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/denim rain/friends_MIST_ROI_SoundNetEncoding_conv_0700515-01-04-03_opt110_20211201-21:07:28.pt",
    # "/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/celestial snow/friends_MIST_ROI_SoundNetEncoding_conv_0700910-01-04-03_opt110_20211201-13:20:33.pt",
    # '/home/maelle/Results/20220126_Hypertraining_analysis/MIST_ROI/celestial snow/friends_MIST_ROI_SoundNetEncoding_conv_0700910-01-04-03_opt110_20211201-15:13:50.pt'
    # ]  

    #selected_r2 = [0.4277, 0.4253, 0.3957, 0.3057,0.3972,0.3584, 0.3673, 0.36, 0.3723, 0.3296]

    result_path = '/home/maelle/Results/20220208_finefriends/20220208_finetuning/sub-03'
    voxels = {'none':[], 'conv7':[], 'conv6':[], 'conv5':[], 'conv4':[]}
    rois = {'none':[], 'conv7':[], 'conv6':[], 'conv5':[], 'conv4':[]}
    for filename in os.listdir(result_path):
        filepath = os.path.join(result_path, filename)     
        name, _ = os.path.splitext(filename)
        finetune_index = name.find('f_conv')
        finetune = name[finetune_index+2:finetune_index+7] if finetune_index > -1 else 'none'
        title = '{}_{}_{}_r2_map'.format(name[:16], finetune, name[-15:])
        data = load(filepath, map_location=device('cpu'))
        #if data['val_r2_max'] in selected_r2 : 
        
        if name.find('Voxels') > -1 :
            voxels[finetune].append(data[criteria]) 
        else :
            rois[finetune].append(data[criteria])
        
    for (vk, vi), (rk, ri) in zip(voxels.items(), rois.items()):
        v_arr = np.array(vi).reshape(len(vi), -1)
        vmoy_arr = np.mean(v_arr)
        vtitle = 'voxels_{}_r2_map'.format(vk)
        voxels_map(vmoy_arr, vtitle, result_path, threshold=0.05)

        r_arr = np.array(ri).reshape(len(ri), -1)
        rmoy_arr = np.mean(r_arr)
        rtitle = 'roi_{}_r2_map'.format(rk)
        ROI_map(rmoy_arr, rtitle, result_path, threshold=0.05)

            # try:
                # voxels_map(filepath, title, result_path, threshold=0.02)
            # except pickle.UnpicklingError :
                # pass
        # else :
            # try : 
                # ROI_map(filepath, title, result_path, threshold=0)
            # except pickle.UnpicklingError :
                # pass  

