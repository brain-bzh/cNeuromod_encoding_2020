import os
import numpy as np
import pandas as pd
import pickle
from torch import load

from files_utils import create_dir_if_needed

from nilearn import image, plotting
from nilearn.plotting import plot_stat_map
from nilearn.regions import signals_to_img_labels
from nilearn.image import load_img, mean_img

from matplotlib import pyplot as plt 


ROI_info = '/home/maelle/Database/MIST_parcellation/Parcel_Information/MIST_ROI.csv'
data_path = '/home/maelle/Results/encoding_12_2020/batch_15'
out_directory = '/home/maelle/Results/encoding_12_2020/analysis/batch_15'
create_dir_if_needed(out_directory)

def plot_train_val_data(criterion, label, data, measure, colors = ['b', 'g', 'm', 'r']) : 
    f = plt.figure()
    legends = []
    for color, (key, data_dict) in zip(colors, data):
        plt.plot(data_dict['train_'+str(measure)], color+'-')
        plt.plot(data_dict['val_'+str(measure)], color+'--')
        legends.append(key+'_Train')
        legends.append(key+'_Val')

    plt.legend(legends, loc='upper right')
    plt.title(str(measure)+' in '+str(criterion))
    f.savefig(os.path.join(out_directory, 'all_{}_{}_in_{}.jpg'.format(label, measure, criterion)))
    plt.close()

def construct_data_dico(criterion, extension, data_path):
    all_data = {}
    key_list = []
    for path, dirs, files in os.walk(data_path):
        for file in files:

            dir_name = os.path.basename(path)
            file_path = os.path.join(path, file)
            name, ext = os.path.splitext(file)

            sub = 'none'
            index = path.find('sub')
            if index != -1 :
                sub = path[index:index+5]

            if criterion == 'sub' :
                key = sub
                value = dir_name 
            else :
                key = dir_name
                value = sub

            if key not in key_list : 
                all_data[key] = []
                key_list.append(key)

            if ext == extension:
                all_data[key].append((value, file_path))
    return all_data
    
all_data = construct_data_dico('film', '.pt', data_path)
all_maps = construct_data_dico('film', '.gz', data_path)

for sub, films in all_data.items():
    all_loaded = [(dir_name, load(file_path)) for (dir_name, file_path) in films]
    all_data[sub] = all_loaded

for sub, films in all_maps.items():
    all_loaded = [(dir_name, image.load_img(file_path)) for (dir_name, file_path) in films]
    all_maps[sub] = all_loaded

#plot
for key, data in all_data.items():
    plot_train_val_data(key, 'subs', data, "loss")
    plot_train_val_data(key, 'subs', data, "r2_max")
    plot_train_val_data(key, 'subs', data, "r2_mean")

#r2 map mean

for sub, films in all_maps.items():
    save = os.path.join(out_directory, str(sub)+'.jpg')
    nifti = [nifti_files for (film_name, nifti_files) in films]
    mean_r2_map = mean_img(nifti)
    plot_stat_map(mean_r2_map, threshold = 0.03, output_file=save)

#BEST ROI ------------------------------------------------------------------
df = pd.read_csv(ROI_info, sep=';', index_col=0)
n = 8
all_index = []
for sub, films_data in all_data.items():
    for (film, data) in films_data:
        r2_by_ROI = data['r2']
        best_index = np.flip(np.argsort(r2_by_ROI))[:n]
        best_index += 1
        all_index.extend(list(best_index))

indexes = set(all_index)
#print(indexes)
labels_ROI = {}
for index in indexes:
    labels_ROI[index] = df['name'][index]

for index, roi in labels_ROI.items():
    pass
    #print(index, roi)

for sub, films_data in all_data.items():
    for (film, data) in films_data:
        r2_by_ROI = data['r2']
        best_index = np.flip(np.argsort(r2_by_ROI))[:n]
        best_index += 1
        #print(best_index, sub, film)
#--------------------------------------------------------------------------------------