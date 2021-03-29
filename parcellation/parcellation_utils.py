#!/usr/bin/env python
# coding: utf-8
# First do a function that does parcellation for one individual file
# These are the files we will need for just one processing

from nilearn.input_data import NiftiMasker
from files_utils import create_dir_if_needed

from load_confounds import Params24
import os,sys
import numpy as np

import pandas as pd 
from matplotlib import pyplot as plt 
from nilearn.plotting import plot_matrix
from nilearn.connectome import ConnectivityMeasure


# basepath = '/home/nfarrugi/git/neuromod/cneuromod/movie10/derivatives/fmriprep1.5.0/fmriprep'
basepath = '~/projects/rrg-pbellec/datasets/cneuromod_new/friends/derivatives/fmriprep-20.1.0/fmriprep/'
mistroicsv = '~/DataBase/fMRI_parcellations/MIST_parcellation/Parcel_Information/MIST_ROI.csv'
mistroi_labelsimg = '~/DataBase/fMRI_parcellations/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'

savepath = '~/DataBase/fMRI_Embeddings/Friends/embed_2021_norm'
create_dir_if_needed(savepath)


def connectome_test(fmri_data, savepath):
    myconn = ConnectivityMeasure(kind='correlation',discard_diagonal=True)
    conn = myconn.fit_transform(fmri_data.reshape(1,406,210))
    conn.shape

    roiinfo = pd.read_csv(mistroicsv,sep=';')
    labels = roiinfo['name'].to_numpy()

    f=plt.figure(figsize=(40,20))
    plt.subplot(2,1,1)
    plot_matrix(conn[0],reorder=True,labels=labels,figure=f)
    plt.subplot(2,1,2)
    plt.hist(conn[0].ravel())
    f.savefig(savepath)


def parcellate_auditory(filepath_fmri,auditorymask,save=False,savepath='./results'):
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
            os.makedirs(savepath,exist_ok=True)
            np.savez_compressed(filepath+'.npz',X=X)
            connectome_test(fmri_data=X, savepath=filepath+'.jpg')
        
        
        return X
    
def parcellate_MIST(filepath_fmri,labels_img=mistroi_labelsimg,save=False,savepath='./results'):
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
            os.makedirs(savepath,exist_ok=True)          
            np.savez_compressed(os.path.join(savepath,filebase+'.npz'),X=X)
            connectome_test(fmri_data=X, savepath=filepath+'.jpg')
        
        return X  


if __name__ == "__main__":

    parser.add_argument("-s", "--subject", type=str,
            help="Specify the subject.")
    args = parser.parse_args()
    subject = "sub-"+args.subject

    subjectdir = os.path.join(basepath, subject)
    savepath = os.path.join(savepath, subject)
    for s in os.walk(subjectdir):
        try:
            curdir = s[0]
            if (curdir[-4:]=='func'):
                print(curdir)
                print('list of files : ')
                
                for curfile in s[2]:
                    curid = (curfile.find('preproc_bold.nii.gz'))
                    
                    if curid >0:
                        print('Parcellating file ' + os.path.join(curdir,curfile))
                        parcellate_MIST(os.path.join(curdir,curfile),save=True,savepath=savepath)
                        parcellate_auditory(os.path.join(curdir,curfile),auditorymask='parcellation/STG_middle.nii.gz',save=True,savepath=savepath)                
        except Exception as e:
            print("Error with file {}".format(curfile))
            print(e)
