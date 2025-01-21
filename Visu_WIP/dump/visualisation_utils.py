import os
import numpy as np
import pandas as pd
import pickle
from torch import load, device

from files_utils import create_dir_if_needed, print_dict, extract_value_from_string

from nilearn import image, plotting, datasets, surface
from nilearn.plotting import plot_stat_map, view_img, view_img_on_surf, plot_matrix
from nilearn.regions import signals_to_img_labels
from nilearn.image import load_img, mean_img
from nilearn.input_data import NiftiMasker
from nilearn.connectome import ConnectivityMeasure

from matplotlib import pyplot as plt 


# roi_path = '/home/maelle/Database/MIST_parcellation/Parcel_Information/MIST_ROI.csv'
# data_path = '/home/maelle/Results/202012_test_seqLen_embed2019'
# target_dir = 'batch_30'
# target_path = os.path.join(data_path, target_dir)
# out_directory = os.path.join(data_path, 'analysis', target_dir)

mistroicsv = '/home/maelle/GitHub_repositories/cNeuromod_encoding_2020/parcellation/MIST_ROI.csv'
roi_path = '/home/maelle/Database/MIST_parcellation/Parcel_Information/MIST_ROI.csv'
datapath = '/home/maelle/Results/20210211_tests_kernel_MIST_ROI_embed_2020_norm/subject_0'
out_directory = os.path.join(datapath, 'analysis')
create_dir_if_needed(out_directory)

def connectome_test(fmri_data, savepath, roi = True):
    myconn = ConnectivityMeasure(kind='correlation',discard_diagonal=True)
    conn = myconn.fit_transform(fmri_data)

    roiinfo = pd.read_csv(mistroicsv,sep=';')
    labels = roiinfo['name'].to_numpy()

    f=plt.figure(figsize=(40,20))
    plt.subplot(2,1,1)
    if roi :
        plot_matrix(conn[0],reorder=True,labels=labels,figure=f)
    else :
        plot_matrix(conn[0],figure=f)
    plt.subplot(2,1,2)
    plt.hist(conn[0].ravel())
    f.savefig(savepath)


#--prep data---------------------------------------------------------------

def construct_data_dico(data_path, extension, criterion='sub', target=None):
    all_data = {}
    key_list = []
    for path, dirs, files in os.walk(data_path):
        for file in files:
            dir_name = os.path.basename(path)
            file_path = os.path.join(path, file)
            name, ext = os.path.splitext(file)

            if ext == extension:
                sub = 'none'
                index = path.find('sub')
                if index != -1 :
                    sub = path[index:index+4]

                if target != None:
                    index = name.find(target)
                    if index != -1:
                        i = index+3 
                        target_value = extract_value_from_string(string=name, start_index = i)
                else :
                    target_value = None

                if criterion == 'sub' :
                    key = sub
                    value = dir_name 
                else :
                    key = dir_name
                    value = sub

                if key not in key_list : 
                    all_data[key] = []
                    key_list.append(key)

                all_data[key].append((value, file_path, (target, target_value)))

    return all_data

def sort_by_target_value(our_data_dico):
    for subject, films_files in our_data_dico.items():
        sorted_index = sorted(set([target_value for (value, file_path, (target, target_value)) in films_files]))
        sorted_list = sorted_index*4

        for (value, file_path, (target, target_value)) in films_files:
            index = sorted_list.index(target_value)
            sorted_list[index] = (value, file_path, (target, target_value))
        our_data_dico[subject] = sorted_list
    return our_data_dico

def subdivise_dict(dico, start_pt = 0):
    for key, value in dico.items():
        start = value[0][start_pt]
        sub_keys = [start]
        for (sub_key, _, _) in value:
            if sub_key != start:
                sub_keys.append(sub_key)
                start = sub_key

        subdivisions = []
        for sub_key in sub_keys:
            subdivision = [sub_key]
            if start_pt == 0:
                subdivision.extend([(b,c) for (a,b,c) in value if a == sub_key])
            elif start_pt == 1:
                subdivision.extend([(a,c) for (a,b,c) in value if b == sub_key])
            subdivisions.append(subdivision)
        dico[key] = subdivisions
    return dico

#BEST ROI ------------------------------------------------------------------
def top_data(data, criteria, n=5):
    selected_values = data[criteria]
    best_values = np.flip(np.sort(selected_values))[:n]
    best_values = np.reshape(best_values, (-1,1))

    best_index = np.flip(np.argsort(selected_values))[:n]
    best_index = np.reshape(best_index, (-1,1))

    best_results = np.concatenate((best_index, best_values), axis=1)
    return best_results

def plot_ROI(outpath, all_data, nROI_shown=5):
    roi_info = roi_path
    df_roi = pd.read_csv(roi_info, sep=';', index_col=0)

    #search for roi/indexes with best r2 values
    all_top_index = []
    all_index = []
    for sub, films_data in all_data.items():
        for (film, data) in films_data:
            result = top_data(data, criteria='r2', n=nROI_shown)
            top_index = result[:,0]+1
            all_top_index.append(list(top_index))
            all_index.extend(list(top_index))
    
    #create a dataframe with info for each roi/index
    indexes = set(all_index)
    stat_ROI = pd.DataFrame()
    for index in indexes:
        label = df_roi['name'][index]
        index_count = [0]*nROI_shown
        for top_index in all_top_index:
            try:
                here = top_index.index(index)
                index_count[here] += 1
            except ValueError:
                pass    
        entry = pd.Series({'label':label, 'count':index_count}, name=int(index))
        stat_ROI = stat_ROI.append(entry)

    #actual plot
    ind = np.arange(nROI_shown)
    bottom = [0]*nROI_shown
    plot_legends = []
    label_legends = []
    for roi, data in stat_ROI.iterrows():

        plot = plt.bar(ind, data['count'], bottom=bottom, tick_label = roi)
        bottom = [i+j for i,j in zip(bottom, data['count'])]

        if sum(data['count']) > 1:
            plot_legends.append(plot[0])
            label_legends.append(data['label'])

    plt.xticks(ind, [str(num+1)+' rank' for num in ind])
    plt.yticks(np.arange(0,max(bottom)+2,2))
    plt.legend(plot_legends, label_legends)

    save_path = os.path.join(outpath, 'best_roi_rank_plot.jpg')
    plt.savefig(save_path)

#-multiple training plot-------------------------------------------------------------------------
def plot_train_val_data(criterion, label, data, measure, colors = ['b', 'g', 'm', 'r']) : 
    f = plt.figure()
    legends = []
    for color, test in zip(colors, data):
        key = test[0]
        data_dict = test[1]
        plt.plot(data_dict['train_'+str(measure)], color+'-')
        plt.plot(data_dict['val_'+str(measure)], color+'--')
        legends.append(key+'_Train')
        legends.append(key+'_Val')

    plt.legend(legends, loc='upper right')
    plt.title(str(measure)+' in '+str(criterion))
    f.savefig(os.path.join(out_directory, 'all_{}_{}_in_{}.jpg'.format(label, measure, criterion)))
    plt.close()

def one_train_plot(criterion, data, measure, colors = ['b', 'g', 'm', 'r']) : 
    legends = []
    #print(f'data_', len(data))
    for color, test in zip(colors, data):
        key = test[0]
        data_dict = test[1]
        #print(f'test_', len(test))
        plt.plot(data_dict['train_'+str(measure)], color+'-')
        plt.plot(data_dict['val_'+str(measure)], color+'--')
        #plt.text(14, 1, 'test_'+str(measure)+' : '+str(data_dict['test_'+str(measure)]))
        legends.append(key+'_Train')
        legends.append(key+'_Val')

    plt.legend(legends, loc='upper right')
    plt.title(str(measure)+' in '+str(criterion))

def brain_3D_map(stat_img, title='', hemishpere='right', threshold=1.0, figure=None, output_file=None):
    fsaverage = datasets.fetch_surf_fsaverage()
    texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)
    plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi=hemishpere,title=title, colorbar=True,threshold=threshold, bg_map=fsaverage.sulc_right,output_file=output_file, figure=figure)
    #plus contour of ROI

# def brain_stat_map()

def multiples_plots(plot_function, criteria, data, measures, out_directory):
    f = plt.figure(figsize=(15*len(measures),15*len(data)))
    for i, film in enumerate(data) : 
        sub_name = film[0]
        target_name = film[1][1][0]
        #---------to correct in script----------------------------------
        sub_data = [(target[0]+'_'+str(target[1]), dico) for (dico, target) in film[1:]]
        #------------------------------------------
        for j, measure in enumerate(measures):
            ax = plt.subplot(len(data),len(measures),len(measures)*i+(j+1))
            #plot_function(sub_name, sub_data, measure)
            
    f.savefig(os.path.join(out_directory, 'all_{}_data_in_{}.jpg'.format(target_name, criteria)))

if __name__ == "__main__":
    target_path = out_directory
    auditorymask='STG_middle.nii.gz'

    all_data = construct_data_dico(data_path=datapath, extension='.pt', criterion='sub', target='ks')
    #all_maps = construct_data_dico('film', '.gz', target_path)
    all_data = sort_by_target_value(all_data)
    for sub, films in all_data.items():
        all_loaded = [(dir_name, load(file_path, map_location=device('cpu')), target_data) for (dir_name, file_path, target_data) in films]
        all_data[sub] = all_loaded

    all_data = subdivise_dict(all_data)

    measures = ["loss", "r2_max", "r2_mean"]
    for subject, data in all_data.items():
        multiples_plots(one_train_plot, subject, data, measures, out_directory)

    # #transform data in nii.gz
    # for subject, value in all_data.items():
    #     for film_data in value:
    #         film_name = film_data[0]
    #         datas = film_data[1:]
    #         for data_target in datas:
    #             target = data_target[1]
    #             training_data = data_target[0]

    #             target_str = target[0]+'_'+str(target[1])

    #             title = 'r2_map_for_{}_{}_{}'.format(target_str, film_name, subject)
    #             map_path = os.path.join(out_directory, title)

    #             mymasker = NiftiMasker(mask_img=auditorymask,standardize=False,detrend=False,t_r=1.49,smoothing_fwhm=8)
    #             mymasker.fit()

    #             r2_stat = mymasker.inverse_transform(training_data['test_r2'].reshape(1,-1))
    #             r2_stat.to_filename(map_path+'.nii.gz')

    #             #brain_3D_map(r2_stat, title=title, hemishpere='right', threshold=0.05, output_file=map_path+'.png')
    #             plot_stat_map(r2_stat, threshold = 0.00, output_file=map_path+'_stat_map.png', title=title)
    


#__________________________________________________  


    # for sub, films in all_maps.items():
    #     all_loaded = [(dir_name, image.load_img(file_path)) for (dir_name, file_path) in films]
    #     all_maps[sub] = all_loaded

    #plot best roi
    #plot_ROI(out_directory, all_data, nROI_shown=5)


    # #plot
    # for key, data in all_data.items(): 
    #     plot_train_val_data(key, 'films', data, "loss")
    #     plot_train_val_data(key, 'films', data, "r2_max")
    #     plot_train_val_data(key, 'films', data, "r2_mean")

    # #r2 map mean
    # for sub, films in all_maps.items():
    #     save = os.path.join(out_directory, str(sub)+'.jpg')
    #     nifti = [nifti_files for (film_name, nifti_files) in films]
    #     mean_r2_map = mean_img(nifti)
    #     plot_stat_map(mean_r2_map, threshold = 0.03, output_file=save)


#previous visualisation code-(go below all test_mist_neuromod_data.py script)-------------------------------------------------------
    #  ### Plot the loss figure
    #     f = plt.figure(figsize=(20,40))

    #     ax = plt.subplot(4,1,2)

    #     plt.plot(state['train_loss'][1:])
    #     plt.plot(state['val_loss'][1:])
    #     plt.legend(['Train','Val'])
    #     plt.title("loss evolution => Mean test R^2=${}, Max test R^2={}, for model {}, batchsize ={} and {} hidden neurons".format(r2model.mean(),r2model.max(), "sdn_1_conv", str(batchsize), fmrihidden))

    #     ### Mean R2 evolution during training
    #     ax = plt.subplot(4,1,3)

    #     plt.plot(state['train_r2_mean'][1:])
    #     plt.plot(state['val_r2_mean'][1:])
    #     plt.legend(['Train','Val'])
    #     plt.title("Mean R^2 evolution for model {}, batchsize ={} and {} hidden neurons".format("sdn_1_conv", str(batchsize), fmrihidden))

    #     ### Max R2 evolution during training
    #     ax = plt.subplot(4,1,4)

    #     plt.plot(state['train_r2_max'][1:])
    #     plt.plot(state['val_r2_max'][1:])
    #     plt.legend(['Train','Val'])
    #     plt.title("Max R^2 evolution for model {}, batchsize ={} and {} hidden neurons".format("sdn_1_conv", str(batchsize), fmrihidden))

    #     ### R2 figure 
    #     #r2_img = signals_to_img_labels(r2model.reshape(1,-1),mistroifile)

    #     #ax = plt.subplot(4,1,1)

    #     #plot_stat_map(r2_img,display_mode='z',cut_coords=8,figure=f,axes=ax)
    #     #f.savefig(str_bestmodel_plot)
    #     #r2_img.to_filename(str_bestmodel_nii)
    #     plt.close()