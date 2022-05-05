from nilearn import datasets, surface, plotting, regions, image, input_data
import os
import matplotlib.pyplot as plt

roi_sub_base = {
    'sub02' : '/home/maelle/Results/sub02_finetune_none_roi_r2_map.nii',
    'sub03' : '/home/maelle/Results/sub03_finetune_none_roi_r2_map.nii',
    'sub04' : '/home/maelle/Results/sub04_finetune_none_roi_r2_map.nii',
    'sub06' : '/home/maelle/Results/sub06_finetune_none_roi_r2_map.nii'
}

roi_sub_conv7 = {
    'sub02' : '/home/maelle/Results/sub02_finetune_conv7_roi_r2_map.nii',
    'sub03' : '/home/maelle/Results/sub03_finetune_conv7_roi_r2_map.nii',
    'sub04' : '/home/maelle/Results/sub04_finetune_conv7_roi_r2_map.nii',
    'sub06' : '/home/maelle/Results/sub06_finetune_conv7_roi_r2_map.nii'
}

roi_sub_conv6 = {
    'sub02' : '/home/maelle/Results/sub02_finetune_conv6_roi_r2_map.nii',
    'sub03' : '/home/maelle/Results/sub03_finetune_conv6_roi_r2_map.nii',
    'sub04' : '/home/maelle/Results/sub04_finetune_conv6_roi_r2_map.nii',
    'sub06' : '/home/maelle/Results/sub06_finetune_conv6_roi_r2_map.nii'
}

roi_sub_conv5 = {
    'sub02' : '/home/maelle/Results/sub02_finetune_conv5_roi_r2_map.nii',
    'sub03' : '/home/maelle/Results/sub03_finetune_conv5_roi_r2_map.nii',
    'sub04' : '/home/maelle/Results/sub04_finetune_conv5_roi_r2_map.nii',
    'sub06' : '/home/maelle/Results/sub06_finetune_conv5_roi_r2_map.nii'
}

roi_sub_conv4 = {
    'sub02' : '/home/maelle/Results/sub02_finetune_conv4_roi_r2_map.nii',
    'sub03' : '/home/maelle/Results/sub03_finetune_conv4_roi_r2_map.nii',
    'sub04' : '/home/maelle/Results/sub04_finetune_conv4_roi_r2_map.nii',
    'sub06' : '/home/maelle/Results/sub06_finetune_conv4_roi_r2_map.nii'
}

roi_sub_base = {
    'sub02' : '/home/maelle/Results/sub02_finetune_none_auditory_voxels_r2_map.nii',
    'sub03' : '/home/maelle/Results/sub03_finetune_none_auditory_voxels_r2_map.nii',
    'sub04' : '/home/maelle/Results/sub04_finetune_none_auditory_voxels_r2_map.nii',
    'sub06' : '/home/maelle/Results/sub06_finetune_none_auditory_voxels_r2_map.nii'
}

auditory_voxels_sub_conv7 = {
    'sub02' : '/home/maelle/Results/sub02_finetune_conv7_auditory_voxels_r2_map.nii',
    'sub03' : '/home/maelle/Results/sub03_finetune_conv7_auditory_voxels_r2_map.nii',
    'sub04' : '/home/maelle/Results/sub04_finetune_conv7_auditory_voxels_r2_map.nii',
    'sub06' : '/home/maelle/Results/sub06_finetune_conv7_auditory_voxels_r2_map.nii'
}

auditory_voxels_sub_conv6 = {
    'sub02' : '/home/maelle/Results/sub02_finetune_conv6_auditory_voxels_r2_map.nii',
    'sub03' : '/home/maelle/Results/sub03_finetune_conv6_auditory_voxels_r2_map.nii',
    'sub04' : '/home/maelle/Results/sub04_finetune_conv6_auditory_voxels_r2_map.nii',
    'sub06' : '/home/maelle/Results/sub06_finetune_conv6_auditory_voxels_r2_map.nii'
}

auditory_voxels_sub_conv5 = {
    'sub02' : '/home/maelle/Results/sub02_finetune_conv5_auditory_voxels_r2_map.nii',
    'sub03' : '/home/maelle/Results/sub03_finetune_conv5_auditory_voxels_r2_map.nii',
    'sub04' : '/home/maelle/Results/sub04_finetune_conv5_auditory_voxels_r2_map.nii',
    'sub06' : '/home/maelle/Results/sub06_finetune_conv5_auditory_voxels_r2_map.nii'
}

auditory_voxels_sub_conv4 = {
    'sub02' : '/home/maelle/Results/sub02_finetune_conv4_auditory_voxels_r2_map.nii',
    'sub03' : '/home/maelle/Results/sub03_finetune_conv4_auditory_voxels_r2_map.nii',
    'sub04' : '/home/maelle/Results/sub04_finetune_conv4_auditory_voxels_r2_map.nii',
    'sub06' : '/home/maelle/Results/sub06_finetune_conv4_auditory_voxels_r2_map.nii'
}

for dicto, nb_conv in zip((roi_sub_conv7, roi_sub_conv6, roi_sub_conv5, roi_sub_conv4), [7,6, 5, 4]) : 
    for (sub, base_path), (sub, conv_path) in zip(roi_sub_base.items(), dicto.items()):
        bp = image.load_img(base_path)
        conv = image.load_img(conv_path)
        contrast_img = image.math_img("conv - bp", conv=conv, bp=bp)
        f = plt.figure()
        plotting.plot_stat_map(contrast_img,display_mode='z' ,cut_coords=6,figure=f, threshold=0, colorbar=True)
        f.savefig(os.path.join('/home/maelle/Results', 'contrast_{}_roi_{}'.format(sub, nb_conv)))
        plt.close()
