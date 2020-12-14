#!/usr/bin/env python
# coding: utf-8
# First do a function that does parcellation for one individual file
# These are the files we will need for just one processing

from nilearn.input_data import NiftiLabelsMasker

from load_confounds import Params24
import os,sys
import numpy as np

import pandas as pd 
from matplotlib import pyplot as plt 
from nilearn.plotting import plot_matrix


basepath = '/home/nfarrugi/git/neuromod/cneuromod/movie10/derivatives/fmriprep1.5.0/fmriprep'

subjectdir = os.path.join(basepath,sys.argv[1])

savepath = '/home/nfarrugi/movie10_parc'

mistroicsv = '/home/nfarrugi/git/MIST_parcellation/MIST_parcellation/Parcel_Information/MIST_ROI.csv'
mistroi_labelsimg = '/home/nfarrugi/git/MIST_parcellation/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'

def parcellate_MIST(filepath_fmri,labels_img=mistroi_labelsimg,save=False,savepath='./results'):
    ## From the filepath_fmri, deduce the maskpath and the tsvfile 
    maskpath_fmri = filepath_fmri[:-19]+'brain_mask.nii.gz'
    tsvfile_fmri = filepath_fmri[:-50] + 'desc-confounds_regressors.tsv'

    idfunc = filepath_fmri.find('/func/')

    filebase = filepath_fmri[idfunc+6:-51]

    ## Initialise the masker

    mymasker = NiftiLabelsMasker(labels_img=labels_img,mask_img=maskpath_fmri,standardize=False,detrend=False,t_r=1.49,smoothing_fwhm=8)

    mymasker.fit()

    # Load the confounds using the Params24 strategy 
    confounds = Params24().load(tsvfile_fmri)
    ## Apply the masker 

    X = mymasker.fit_transform(filepath_fmri,confounds=confounds)
    
    if save:
        os.makedirs(savepath,exist_ok=True)
        
        np.savez_compressed(os.path.join(savepath,filebase+'npz'),X=X)
    
    return X
    
    

for s in os.walk(subjectdir):
    curdir = s[0]
    try:
        if (curdir[-4:]=='func'):
            print(curdir)
            print('list of files : ')
            
            for curfile in s[2]:
                curid = (curfile.find('preproc_bold.nii.gz'))
                
                if curid >0:
                    print('Parcellating file ' + os.path.join(curdir,curfile))
                    X = parcellate_MIST(os.path.join(curdir,curfile),save=True,savepath=savepath)
    except:
        print("error with file {}".format(curfile))


# Connectome tests
# --

if False:
    from nilearn.connectome import ConnectivityMeasure

    myconn = ConnectivityMeasure(kind='correlation',discard_diagonal=True)

    conn = myconn.fit_transform(X.reshape(1,406,210))

    conn.shape

    roiinfo = pd.read_csv(mistroicsv,sep=';')

    labels = roiinfo['name'].to_numpy()

    f=plt.figure(figsize=(20,20))
    plot_matrix(conn[0],labels=labels,figure=f)

    from nilearn.plotting import plot_matrix
    get_ipython().run_line_magic('matplotlib', 'inline')
    f=plt.figure(figsize=(20,20))
    plot_matrix(conn[0],reorder=True,labels=labels,figure=f)

    plt.hist(conn[0].ravel())
