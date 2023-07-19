import os
import numpy as np
from sklearn import preprocessing

from files_utils import create_dir_if_needed

path_parcellation = '/home/maelle/Database/cNeuromod_Embeddings/embed_2020/MIST_ROI'
save_path = '/home/maelle/Database/cNeuromod_Embeddings/embed_2020_norm/MIST_ROI'

standardize = preprocessing.StandardScaler()
for subject in os.listdir(path_parcellation):
    subject_path = os.path.join(path_parcellation, subject)
    save_sub_path = os.path.join(save_path, subject)
    create_dir_if_needed(save_sub_path)

    for run in os.listdir(subject_path):
        run_path = os.path.join(subject_path, run)
        save_run_path = os.path.join(save_sub_path, run)
        X =standardize.fit_transform(np.load(run_path)['X'].T).T

        np.savez_compressed(save_run_path, X=X)






