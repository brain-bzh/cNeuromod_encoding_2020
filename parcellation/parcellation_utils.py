#!/usr/bin/env python
# coding: utf-8
# First do a function that does parcellation for one individual file
# These are the files we will need for just one processing

from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

from load_confounds import Params24
import os,sys, argparse
import numpy as np

import pandas as pd 
from matplotlib import pyplot as plt 
from nilearn.plotting import plot_matrix
from nilearn.connectome import ConnectivityMeasure


# basepath = '/~/nfarrugi/git/neuromod/cneuromod/movie10/derivatives/fmriprep1.5.0/fmriprep'
basepath = '/home/maellef/projects/rrg-pbellec/datasets/cneuromod_new/friends/derivatives/fmriprep-20.1.0/fmriprep/'
mistroicsv = '/home/maellef/DataBase/fMRI_parcellations/MIST_parcellation/Parcel_Information/MIST_ROI.csv'
mistroi_labelsimg = '/home/maellef/DataBase/fMRI_parcellations/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'
auditory_mask = '/home/maellef/git_dir/cNeuromod_encoding_2020/parcellation/STG_middle.nii.gz'
savepath = '/home/maellef/DataBase/fMRI_Embeddings/Friends/embed_2021_norm'
os.makedirs(savepath,exist_ok=True)

def parcellate_auditory(filepath_fmri,auditorymask,save=True,savepath='./results'):
    savepath = os.path.join(savepath, 'auditory_Voxels')
    ## From the filepath_fmri, deduce the maskpath and the tsvfile 
    tsvfile_fmri = filepath_fmri[:-50] + 'desc-confounds_regressors.tsv'
    idfunc = filepath_fmri.find('/func/')
    filebase = filepath_fmri[idfunc+6:-51]
    filepath = os.path.join(savepath,filebase)
    print(filepath)
    if os.path.isfile(filepath):
        print("File {} already exists".format(filepath))
        return np.load(filepath)['X']
    else:

        ## Initialise the maslabels_img=labels_img,ker

        mymasker = NiftiMasker(mask_img=auditorymask,standardize=True,detrend=False,t_r=1.49,smoothing_fwhm=8)

        mymasker.fit()

        # Load the confounds using the Params24 strategy 
        confounds = Params24().load(tsvfile_fmri)
        ## Apply the masker 

        X = mymasker.fit_transform(filepath_fmri,confounds=confounds)
        
        if save:
            print('saving voxel parcellations ...')
            os.makedirs(savepath,exist_ok=True)
            np.savez_compressed(filepath+'.npz',X=X)        
        
        return X
    
def parcellate_MIST(filepath_fmri,labels_img=mistroi_labelsimg,save=True,savepath='./results'):
    savepath = os.path.join(savepath, 'MIST_ROI')
    ## From the filepath_fmri, deduce the maskpath and the tsvfile 
    maskpath_fmri = filepath_fmri[:-19]+'brain_mask.nii.gz'
    tsvfile_fmri = filepath_fmri[:-50] + 'desc-confounds_regressors.tsv'

    idfunc = filepath_fmri.find('/func/')

    filebase = filepath_fmri[idfunc+6:-51]
    filepath = os.path.join(savepath,filebase)
    if os.path.isfile(filepath):
        print("File {} already exists".format(filepath))
        return np.load(filepath)['X']
    else:

        ## Initialise the masker

        mymasker = NiftiLabelsMasker(labels_img=labels_img,mask_img=maskpath_fmri,standardize=True,detrend=False,t_r=1.49,smoothing_fwhm=8)

        mymasker.fit()

        # Load the confounds using the Params24 strategy 
        confounds = Params24().load(tsvfile_fmri)
        ## Apply the masker 

        X = mymasker.fit_transform(filepath_fmri,confounds=confounds)
        
        if save:
            print('saving ROI parcellations ...')
            os.makedirs(savepath,exist_ok=True)          
            np.savez_compressed(os.path.join(savepath,filebase+'.npz'),X=X)
        
        return X  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=str, help="Specify the subject.")
    args = parser.parse_args()
    subject = "sub-"+args.subject

    subjectdir = os.path.join(basepath, subject)
    savepath = os.path.join(savepath, subject)
    for s in os.listdir(subjectdir):
        if s.find('ses')>-1:
            sesspath = os.path.join(subjectdir, s, 'func')
            for curfile in os.listdir(sesspath):               
                if curfile.find('preproc_bold.nii.gz')>-1 and curfile.find('2009')>-1:
                    filepath = os.path.join(sesspath, curfile)
                    print('Parcellating file ' + filepath)
                    parcellate_MIST(filepath,save=True,savepath=savepath)
                    parcellate_auditory(filepath,auditorymask=auditory_mask,save=True,savepath=savepath)
