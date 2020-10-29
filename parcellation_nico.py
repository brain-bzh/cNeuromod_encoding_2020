#!/usr/bin/env python
# coding: utf-8

# First do a function that does parcellation for one individual file

# These are the files we will need for just one processing

# In[1]:


fmripath = '/home/nfarrugi/git/neuromod/phantom_video/derivatives/fmriprep/sub-01/ses-vid006/func/sub-01_ses-vid006_task-life_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
maskpath = '/home/nfarrugi/git/neuromod/phantom_video/derivatives/fmriprep/sub-01/ses-vid006/func/sub-01_ses-vid006_task-life_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
tsvfile = '/home/nfarrugi/git/neuromod/phantom_video/derivatives/fmriprep/sub-01/ses-vid006/func/sub-01_ses-vid006_task-life_run-01_desc-confounds_regressors.tsv'

labels_img = '/home/nfarrugi/git/MIST_parcellation/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'


# In[3]:


from nilearn.input_data import NiftiLabelsMasker
import confound_loader
import os
import numpy as np

import pandas as pd 
from matplotlib import pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

from nilearn.plotting import plot_matrix


# In[3]:


def parcellate_MIST(filepath_fmri,labels_img='/home/brain/Data_Base/MIST_parcellation/Parcellations/MIST_ROI.nii.gz',save=False,savepath='./results'):
    ## From the filepath_fmri, deduce the maskpath and the tsvfile 
    maskpath_fmri = filepath_fmri[:-19]+'brain_mask.nii.gz'
    tsvfile_fmri = filepath_fmri[:-50] + 'desc-confounds_regressors.tsv'

    idfunc = filepath_fmri.find('/func/')

    filebase = filepath_fmri[idfunc+6:-51]

    ## Initialise the masker

    mymasker = NiftiLabelsMasker(labels_img=labels_img,mask_img=maskpath_fmri,standardize=True,detrend=True,t_r=1.49)

    mymasker.fit()

    # Load the confounds, including PCA calculation of the motion params 

    confounds_pca = confound_loader.load_confounds(tsvfile_fmri, strategy=["motion","high_pass_filter"], motion_model="6params", n_components=0.95)

    ## Apply the masker 

    X = mymasker.fit_transform(filepath_fmri,confounds=confounds_pca.to_numpy())
    
    if save:
        os.makedirs(savepath,exist_ok=True)
        
        np.savez_compressed(os.path.join(savepath,filebase+'npz'),X=X)
    
    return X
    
    


# In[4]:


os.listdir('/media/brain/Elec_HD/cneuromod/movie10/derivatives/fmriprep1.5.0/fmriprep/sub-03')


# In[5]:


for s in os.walk('/media/brain/Elec_HD/cneuromod/movie10/derivatives/fmriprep1.5.0/fmriprep/sub-03'):
    curdir = s[0]
    if (curdir[-4:]=='func'):
        #print(curdir)
        #print('list of files : ')
        
        for curfile in s[2]:
            curid = (curfile.find('preproc_bold.nii.gz'))
            
            if curid >0:
                print('Parcellating file ' + os.path.join(curdir,curfile))
                X = parcellate_MIST(os.path.join(curdir,curfile),save=True,savepath='/home/brain/Data_Base/movie10_parc/')


# Connectome tests
# --


from nilearn.connectome import ConnectivityMeasure

myconn = ConnectivityMeasure(kind='correlation',discard_diagonal=True)

conn = myconn.fit_transform(X.reshape(1,406,210))

conn.shape

roiinfo = pd.read_csv('/home/nfarrugi/git/MIST_parcellation/MIST_parcellation/Parcel_Information/MIST_ROI.csv',sep=';')

labels = roiinfo['name'].to_numpy()

f=plt.figure(figsize=(20,20))
plot_matrix(conn[0],labels=labels,figure=f)

from nilearn.plotting import plot_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
f=plt.figure(figsize=(20,20))
plot_matrix(conn[0],reorder=True,labels=labels,figure=f)

plt.hist(conn[0].ravel())
