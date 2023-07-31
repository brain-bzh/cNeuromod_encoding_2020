#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np 
from nilearn.image import math_img
from nilearn.masking import intersect_masks

roi_info=pd.read_csv('MIST_ROI.csv',sep=';')

# Fetch just one region


labels_img = 'MIST_ROI.nii.gz'

#selectrois = [154,153,170,171]

selectrois = [154,153]

roi_imgs = []
for roinum in selectrois:
    roi_name = roi_info[roi_info['roi'] == roinum]['name']
    print(roi_name)
    
    roi_imgs.append(math_img(formula='img=={}'.format(roinum),img=labels_img))

allrois = intersect_masks(roi_imgs,threshold=0,connected=False)


allrois.to_filename('STG_middle.nii.gz')




