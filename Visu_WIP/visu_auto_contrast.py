import os
import matplotlib
import matplotlib.pyplot as plt
from nilearn import datasets, surface, plotting, regions, image, input_data
from visu_utils import brain_3D_map

font = {'size'   : 22}

results_path = '/home/maelle/Results/figures_finefriends/nii'
subs = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06']
out_directory = '/home/maelle/Results/figures_finefriends'
relative = True

def difference_nii_image(a_nii_path, b_nii_path, relative = True):
    a = image.load_img(a_nii_path)
    b = image.load_img(b_nii_path)

    if relative:
        a = image.math_img("a+(1**-10000)", a=a) 
        b = image.math_img("b+(1**-10000)", b=b)
        formula = "(a - b)*100/(b)"
    else : 
        formula = "a - b"

    difference_img = image.math_img(formula, a=a, b=b)
    return difference_img

for sub in subs:
    print(sub)
    MIST_roi = {}
    auditory_voxels = {}
    for nii_file in os.listdir(results_path):
        nii_path = os.path.join(results_path, nii_file)
        if sub in nii_file and 'roi' in nii_file:
            if nii_file.find('none') > -1:
                MIST_roi['none'] = nii_path
            elif nii_file.find('conv') > -1:
                i = nii_file.find('conv')
                conv_layer = nii_file[i:i+5]
                MIST_roi[conv_layer] = nii_path

        if sub in nii_file and 'auditory_voxels' in nii_file:
            if nii_file.find('none') > -1:
                auditory_voxels['none'] = nii_path
            elif nii_file.find('conv') > -1:
                i = nii_file.find('conv')
                conv_layer = nii_file[i:i+5]
                auditory_voxels[conv_layer] = nii_path

    if len(MIST_roi)>0 : 
        fig = plt.figure(figsize=(20, 4*4))
        no_ft = MIST_roi['none']
        for i, layer in enumerate([4,5,6,7]):
            matplotlib.rc('font', **font)
            ax = plt.subplot(4,1,i+1)
            difference_image = difference_nii_image(a_nii_path= MIST_roi['conv{}'.format(layer)], b_nii_path=no_ft, relative=relative)
            vmax = 8 if relative else 0.1
            a = plotting.plot_stat_map(difference_image, title=sub, display_mode='z', axes=ax, cut_coords=[-11, 4, 16, 28, 40], colorbar=True, symmetric_cbar=True, threshold=0, vmax=vmax)
            cbar = a._cbar
            print(type(cbar))
        #plt.subplots_adjust(left=0.00, bottom=0.0, right=0.1, top=0.1, wspace=0.0, hspace=0.0)
        #matplotlib.colorbar.ColorbarBase([19, 1, 0.2, 0.8], orientation='vertical')
        #cbar.[19, 1, 0.2, 0.8], orientation='vertical')
        fig.savefig(os.path.join(out_directory, 'contrast_img_{}.png'.format(sub)))
        plt.close()
    
    if len(auditory_voxels)>0:
        print('bloup')

