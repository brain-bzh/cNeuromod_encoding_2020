import os
import numpy as np
from visualisation_utils import connectome_test

mist_roi = '/home/maelle/GitHub_repositories/cNeuromod_encoding_2020/parcellation/MIST_ROI.csv'
auditory_dir = '/home/maelle/these/DataBase/fMRI_Embeddings/Friends/embed_2021_norm/sub-06/auditory_Voxels'
roi_dir = '/home/maelle/these/DataBase/fMRI_Embeddings/Friends/embed_2021_norm/sub-06/MIST_ROI'

result_auditory_dir = os.path.join(auditory_dir, 'result')
result_roi_dir = os.path.join(roi_dir, 'result')

os.makedirs(result_auditory_dir, exist_ok=True)
os.makedirs(result_roi_dir, exist_ok=True)

for filepath_voxel, filepath_roi in zip(os.listdir(auditory_dir), os.listdir(roi_dir)):
    all_path_roi = os.path.join(roi_dir, filepath_roi)
    x_roi = np.load(all_path_roi)['X']
    x_roi = x_roi.reshape(1,-1,210)
    x_roi.shape
    file_save = os.path.join(result_roi_dir, filepath_roi[:-4])
    connectome_test(x_roi, file_save)

    all_path_voxel = os.path.join(auditory_dir, filepath_voxel)
    x_voxel = np.load(all_path_voxel)['X']
    x_voxel = x_voxel.reshape(1,-1,556)
    x_voxel.shape
    file_save = os.path.join(result_auditory_dir, filepath_voxel[:-4])
    connectome_test(x_voxel, file_save, roi=False)





