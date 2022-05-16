from nilearn import datasets, surface, plotting, regions, image, input_data
import os
import matplotlib.pyplot as plt

outpath = '/home/maelle/Results/figures_finefriends2'
os.makedirs(outpath, exist_ok=True)
roi = [
{
    'sub02' : '/home/maelle/Results/figures_finefriends/nii/sub02_finetune_none_roi_r2_map.nii',
    'sub03' : '/home/maelle/Results/figures_finefriends/nii/sub03_finetune_none_roi_r2_map.nii',
    'sub04' : '/home/maelle/Results/figures_finefriends/nii/sub04_finetune_none_roi_r2_map.nii',
    'sub06' : '/home/maelle/Results/figures_finefriends/nii/sub06_finetune_none_roi_r2_map.nii'
},
{
    'sub02' : '/home/maelle/Results/figures_finefriends/nii/sub02_finetune_conv7_roi_r2_map.nii',
    'sub03' : '/home/maelle/Results/figures_finefriends/nii/sub03_finetune_conv7_roi_r2_map.nii',
    'sub04' : '/home/maelle/Results/figures_finefriends/nii/sub04_finetune_conv7_roi_r2_map.nii',
    'sub06' : '/home/maelle/Results/figures_finefriends/nii/sub06_finetune_conv7_roi_r2_map.nii'
},
{
    'sub02' : '/home/maelle/Results/figures_finefriends/nii/sub02_finetune_conv6_roi_r2_map.nii',
    'sub03' : '/home/maelle/Results/figures_finefriends/nii/sub03_finetune_conv6_roi_r2_map.nii',
    'sub04' : '/home/maelle/Results/figures_finefriends/nii/sub04_finetune_conv6_roi_r2_map.nii',
    'sub06' : '/home/maelle/Results/figures_finefriends/nii/sub06_finetune_conv6_roi_r2_map.nii'
},
{
    'sub02' : '/home/maelle/Results/figures_finefriends/nii/sub02_finetune_conv5_roi_r2_map.nii',
    'sub03' : '/home/maelle/Results/figures_finefriends/nii/sub03_finetune_conv5_roi_r2_map.nii',
    'sub04' : '/home/maelle/Results/figures_finefriends/nii/sub04_finetune_conv5_roi_r2_map.nii',
    'sub06' : '/home/maelle/Results/figures_finefriends/nii/sub06_finetune_conv5_roi_r2_map.nii'
},
{
    'sub02' : '/home/maelle/Results/figures_finefriends/nii/sub02_finetune_conv4_roi_r2_map.nii',
    'sub03' : '/home/maelle/Results/figures_finefriends/nii/sub03_finetune_conv4_roi_r2_map.nii',
    'sub04' : '/home/maelle/Results/figures_finefriends/nii/sub04_finetune_conv4_roi_r2_map.nii',
    'sub06' : '/home/maelle/Results/figures_finefriends/nii/sub06_finetune_conv4_roi_r2_map.nii'
}]

voxels = [
{
    'sub02' : './nii/sub02_finetune_none_auditory_voxels_r2_map.nii',
    'sub03' : './nii/sub03_finetune_none_auditory_voxels_r2_map.nii',
    'sub04' : './nii/sub04_finetune_none_auditory_voxels_r2_map.nii',
    'sub06' : './nii/sub06_finetune_none_auditory_voxels_r2_map.nii'
},
{
    'sub02' : './nii/sub02_finetune_conv7_auditory_voxels_r2_map.nii',
    'sub03' : './nii/sub03_finetune_conv7_auditory_voxels_r2_map.nii',
    'sub04' : './nii/sub04_finetune_conv7_auditory_voxels_r2_map.nii',
    'sub06' : './nii/sub06_finetune_conv7_auditory_voxels_r2_map.nii'
},
{
    'sub02' : './nii/sub02_finetune_conv6_auditory_voxels_r2_map.nii',
    'sub03' : './nii/sub03_finetune_conv6_auditory_voxels_r2_map.nii',
    'sub04' : './nii/sub04_finetune_conv6_auditory_voxels_r2_map.nii',
    'sub06' : './nii/sub06_finetune_conv6_auditory_voxels_r2_map.nii'
},
{
    'sub02' : './nii/sub02_finetune_conv5_auditory_voxels_r2_map.nii',
    'sub03' : './nii/sub03_finetune_conv5_auditory_voxels_r2_map.nii',
    'sub04' : './nii/sub04_finetune_conv5_auditory_voxels_r2_map.nii',
    'sub06' : './nii/sub06_finetune_conv5_auditory_voxels_r2_map.nii'
},
{
    'sub02' : './nii/sub02_finetune_conv4_auditory_voxels_r2_map.nii',
    'sub03' : './nii/sub03_finetune_conv4_auditory_voxels_r2_map.nii',
    'sub04' : './nii/sub04_finetune_conv4_auditory_voxels_r2_map.nii',
    'sub06' : './nii/sub06_finetune_conv4_auditory_voxels_r2_map.nii'
}]

def brain_3D_map(stat_img, title='', hemisphere='right', threshold=1.0, figure=None, output_file=None):
    fsaverage = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)
    plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi=hemisphere,title=title, colorbar=True,threshold=threshold, bg_map=fsaverage.sulc_right,output_file=output_file, figure=figure)

for i, scale in zip(['roi'], [roi]):
    database_noft = scale[0]
    database_ft = scale[1:]
    for dicto, nb_conv in zip(database_ft, [7,6, 5, 4]) : 
        for (sub, base_path), (sub, conv_path) in zip(database_noft.items(), dicto.items()):
            bp = image.load_img(base_path)
            conv = image.load_img(conv_path)
            contrast_img = image.math_img("conv - bp", conv=conv, bp=bp)
            out_directory = os.path.join(outpath, '{}_{}_{}'.format(sub, i, nb_conv))
            f = plt.figure(figsize=(8.0,5.5))
            if i == 'roi':
                plotting.plot_stat_map(conv, display_mode='z', cut_coords=[-11, 4, 16, 28, 40], figure=f, colorbar=True, symmetric_cbar=True, threshold=0.05, vmax=0.4)
            else :
                brain_3D_map(contrast_img, hemisphere='right', threshold=0.05, output_file=os.path.join(out_directory))
            f.savefig(out_directory)
            plt.close()
