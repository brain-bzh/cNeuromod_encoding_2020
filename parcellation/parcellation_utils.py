#!/usr/bin/env python
# coding: utf-8
# First do a function that does parcellation for one individual file
# These are the files we will need for just one processing

from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

from load_confounds import Minimal
import os,sys, argparse
import numpy as np

import pandas as pd 
from matplotlib import pyplot as plt 
from nilearn.plotting import plot_matrix
from nilearn.connectome import ConnectivityMeasure


# basepath = '/~/nfarrugi/git/neuromod/cneuromod/movie10/derivatives/fmriprep1.5.0/fmriprep'
cNeuromod_path = '/home/maellef/projects/rrg-pbellec/datasets/cneuromod_new/'
mistroicsv = '/home/maellef/projects/rrg-pbellec/maellef/data/DataBase/fMRI_parcellations/MIST_parcellation/Parcel_Information/MIST_ROI.csv'
mistroi_labelsimg = '/home/maellef/projects/rrg-pbellec/maellef/data/DataBase/fMRI_parcellations/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'
auditory_mask = '/home/maellef/git_dir/cNeuromod_encoding_2020/parcellation/STG_middle.nii.gz'
embedding_path = '/home/maellef/projects/rrg-pbellec/maellef/data/DataBase/fMRI_Embeddings_fmriprep-20.2lts/'
os.makedirs(embedding_path,exist_ok=True)

def parcellate_auditory(filepath_fmri, auditorymask, subject, dataset, save=True,savepath='./results'):
    savepath = os.path.join(savepath, 'auditory_Voxels', dataset, subject)
    ## From the filepath_fmri, deduce the maskpath and the tsvfile 
    tsvfile_fmri = filepath_fmri[:-50] + 'desc-confounds_regressors.tsv'
    idstart = filepath_fmri.find('/func/')
    idend = filepath_fmri.find('_space')
    filebase = filepath_fmri[idstart+6:idend]
    filepath = os.path.join(savepath, filebase+'.npz')
    print(f"parcellation file is ", filebase)

    if os.path.isfile(filepath):
        print("File {} already exists".format(filepath))
        return np.load(filepath)['X']
    else:
        ## Initialise the maslabels_img=labels_img,ker
        mymasker = NiftiMasker(mask_img=auditorymask,standardize=True,detrend=False,t_r=1.49,smoothing_fwhm=8)
        mymasker.fit()
        # Load the confounds using the Params24 strategy 
        confounds = Minimal().load(filepath_fmri)
        ## Apply the masker 
        X = mymasker.fit_transform(filepath_fmri,confounds=confounds)
        
        if save:
            print('saving voxel parcellations ...')
            os.makedirs(savepath,exist_ok=True)
            np.savez_compressed(filepath,X=X)        
        return X
    
def parcellate_MIST(filepath_fmri, subject, dataset, labels_img=mistroi_labelsimg,save=True,savepath='./results'):
    savepath = os.path.join(savepath, 'MIST_ROI', dataset, subject)
    ## From the filepath_fmri, deduce the maskpath and the tsvfile 
    maskpath_fmri = filepath_fmri[:-19]+'brain_mask.nii.gz'
    tsvfile_fmri = filepath_fmri[:-50] + 'desc-confounds_regressors.tsv'
    idstart = filepath_fmri.find('/func/')
    idend = filepath_fmri.find('_space')
    filebase = filepath_fmri[idstart+6:idend]
    filepath = os.path.join(savepath, filebase+'.npz')
    print(f"parcellation file is ", filebase)

    if os.path.isfile(filepath):
        print("File {} already exists".format(filepath))
        return np.load(filepath)['X']
    else:
        ## Initialise the masker
        mymasker = NiftiLabelsMasker(labels_img=labels_img,mask_img=maskpath_fmri,standardize=True,detrend=False,t_r=1.49,smoothing_fwhm=8)
        mymasker.fit()
        # Load the confounds using the Params24 strategy 
        confounds = Minimal().load(filepath_fmri)
        ## Apply the masker 
        X = mymasker.fit_transform(filepath_fmri,confounds=confounds)
        
        if save:
            print('saving ROI parcellations ...')
            os.makedirs(savepath,exist_ok=True)          
            np.savez_compressed(filepath,X=X)
        return X  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=str, help="Specify the subject among 01, 02, 03, 04, 05, and 06.")
    parser.add_argument("-d", "--dataset", type=str, help="Specify the dataset among friends and movie10.")
    args = parser.parse_args()
    subject = "sub-"+args.subject
    dataset = args.dataset
    mri_path = os.path.join(cNeuromod_path, dataset+'/derivatives/fmriprep-20.2lts/fmriprep/')
    subjectdir = os.path.join(mri_path, subject)

    for s in os.listdir(subjectdir):
        if s.find('ses')>-1:
            sesspath = os.path.join(subjectdir, s, 'func')
            for curfile in os.listdir(sesspath):               
                if curfile.find('preproc_bold.nii.gz')>-1 and curfile.find('2009')>-1:
                    filepath = os.path.join(sesspath, curfile)
                    print('Parcellating file ' + filepath)
                    parcellate_MIST(filepath,save=True,savepath=embedding_path, subject=subject, dataset=dataset)
                    parcellate_auditory(filepath,auditorymask=auditory_mask,save=True,savepath=embedding_path, subject=subject, dataset=dataset)
