from matplotlib import pyplot as plt
import files_utils as fu
import os
from torch import load, device
from nilearn import image, plotting, datasets, surface
from nilearn.input_data import NiftiMasker

def one_train_plot_TEST(criterion, data, measure, colors = ['b', 'g', 'm', 'r']) : 
    legends = []

    for color, test in zip(colors, data):
        key = test[0]
        data_dict = test[1]
        plt.plot(data_dict['train_'+str(measure)], color+'-')
        plt.plot(data_dict['val_'+str(measure)], color+'--')
        
        if measure == "loss":
            value =  data_dict['test_'+str(measure)][0]
        else :
            value = data_dict['test_'+str(measure)]
        legends.append(key+'_Train')
        legends.append(key+'_Val')

    plt.legend(legends, loc='upper left', bbox_to_anchor=(1,1))
    plt.title(str(measure)+' in '+str(criterion)+', test_'+str(measure)+' : '+str(value))

if __name__ == "__main__":
    datapath = '/home/maelle/Results/20210216_weightDecay_MIST_ROI210_lr0.01_100epochs_embed2020norm/sub1/bourne_supremacy'
    targets = ['ks', 'wd']
    out_directory = '/home/maelle/Results/analysis_212102_Nicolas/wd'
    prefix = '2020_MIST_ROI210_lr10e-3_sub0_bourne_100epochs'
    fu.create_dir_if_needed(out_directory)

    all_data = []
    all_maps = []
    for data_file in os.listdir(datapath):
        file_path = os.path.join(datapath, data_file)
        basename, ext = os.path.splitext(file_path)
        if ext != '' : 
            targets_values = []
            for target in targets:
                tgt_idx = basename.find(target)
                tgt_value = fu.extract_value_from_string(basename, tgt_idx+len(target), stop_condition=(lambda x: (x=='2')))
                targets_values.append(target+'_'+str(tgt_value))  
            if ext == '.pt':
                data = load(file_path, map_location=device('cpu'))
                data = (targets_values, data)
                all_data.append(data)
            elif ext == '.gz':
                data_map = image.load_img(file_path)
                data_map = (targets_values, data_map)
                all_maps.append(data_map)

measures = ["loss", "r2_max", "r2_mean"]
j=0
first_ks = None

for data in all_data:
    if data[0][0] != first_ks : 
        first_ks = data[0][0]

        data_ks = [(wd, data) for ([ks, wd], data) in all_data if ks==first_ks]
        f = plt.figure(figsize=(5*len(measures)+7, 5))
        for i, measure in enumerate(measures):
            ax = plt.subplot(1,len(measures),i+1)
            one_train_plot_TEST(first_ks, data_ks, measure)

        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.0)
        f.savefig(os.path.join(out_directory, prefix+'training_data_{}'.format(first_ks)))
        plt.close()



#--------------------------------------------------------------------------------------------------------------------------
        ks_map = [(wd, data) for ([ks, wd], data) in all_maps if ks==first_ks]
        f2 = plt.figure(figsize=(10,15))
        for j, data_map in enumerate(ks_map):
            r2_img = data_map[1]
            target = data_map[0]
            ax = plt.subplot(4,1,j+1)
            title = 'R2 Map in bourne : '+str(target)
            plotting.plot_stat_map(r2_img,display_mode='z',cut_coords=5,figure=f2,axes=ax, title=title, threshold=0.1)

        f2.savefig(os.path.join(out_directory, prefix+'{}_r2_maps'.format(first_ks)))
        plt.close()

# #-------------------------------------------------------------------------------------------------------------------------
#         ks_map = [(wd, data) for ([ks, wd], data) in all_maps if ks==first_ks]
#         auditorymask='STG_middle.nii.gz'
#         f3 = plt.figure(figsize=(10,10))
#         for j, training_data in enumerate(ks_map):
#             ax = plt.subplot(4,1,j+1)
#             target = training_data[0]
#             data = training_data[1]
#             title = 'R2 Map in bourne : '+str(target)

#             mymasker = NiftiMasker(mask_img=auditorymask,standardize=False,detrend=False,t_r=1.49,smoothing_fwhm=8)
#             mymasker.fit()

#             r2_stat = mymasker.inverse_transform(data['test_r2'].reshape(1,-1))

#             #brain_3D_map(r2_stat, title=title, hemishpere='right', threshold=0.05, output_file=map_path+'.png')
#             plotting.plot_stat_map(r2_stat, threshold = 0.02, title=title, figure=f3, axes=ax)
#         f3.savefig(out_directory+'/'+prefix+'_stat_map_AudVox.png')
#         plt.close()

