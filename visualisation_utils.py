import os
import numpy as np
import pandas as pd
import pickle
from torch import load

from nilearn import image, plotting
from nilearn.plotting import plot_stat_map
from nilearn.regions import signals_to_img_labels

from matplotlib import pyplot as plt 


ROI_info = '/home/maelle/Database/MIST_parcellation/Parcel_Information/MIST_ROI.csv'
data_path = '/home/maelle/Results/encoding_12_2020/'
out_directory = '/home/maelle/Results/encoding_12_2020/analysis'


def plot_train_val_data(subject, data_films, measure, colors = ['b', 'g', 'm', 'r']) : 
    f = plt.figure()
    legends = []
    for color, (key, data_dict) in zip(colors, data_films):
        plt.plot(data_dict['train_'+str(measure)], color+'-')
        plt.plot(data_dict['val_'+str(measure)], color+'--')
        legends.append(key+'_Train')
        legends.append(key+'_Val')

    plt.legend(legends, loc='upper right')
    plt.title(str(measure)+' in '+str(subject))
    f.savefig(os.path.join(out_directory, 'all_films_{}_in_{}.jpg'.format(measure, subject)))
    plt.close()

all_data = {}
all_maps = {}
previous_sub = 'none'
for path, dirs, files in os.walk(data_path):
    for file in files:

        index = path.find('sub')
        if index != -1 :
            sub = path[index:index+5]
        if sub!=previous_sub : 
            all_data[sub] = []
            all_maps[sub] = []
            previous_sub = sub

        dir_name = os.path.basename(path)
        file_path = os.path.join(path, file)

        name, ext = os.path.splitext(file)

        if ext == '.pt':
            all_data[sub].append((dir_name, file_path))
        elif ext == '.gz':
            all_maps[sub].append((dir_name, file_path))

for sub, films in all_data.items():
    all_loaded = [(dir_name, load(file_path)) for (dir_name, file_path) in films]
    all_data[sub] = all_loaded

for sub, films in all_maps.items():
    all_loaded = [(dir_name, image.load_img(file_path)) for (dir_name, file_path) in films]
    all_maps[sub] = all_loaded


df = pd.read_csv(ROI_info, sep=';', index_col=0)
print(df['name'][1])

n = 10
all_index = []
for sub, films_data in all_data.items():
    for (film, data) in films_data:
        r2_by_ROI = data['r2']
        best_index = np.flip(np.argsort(r2_by_ROI))[:n]
        best_index += 1
        all_index.extend(list(best_index))

indexes = set(all_index)
print(indexes)
labels_ROI = {}
for index in indexes:
    labels_ROI[index] = df['name'][index]

for index, roi in labels_ROI.items():
    print(index, roi)

for sub, films_data in all_data.items():
    for (film, data) in films_data:
        r2_by_ROI = data['r2']
        best_index = np.flip(np.argsort(r2_by_ROI))[:n]
        best_index += 1
        print(best_index, sub, film)





#for sub, data in all_data.items():
    #plot_train_val_data(sub, data, "loss")
    #plot_train_val_data(sub, data, "r2_max")
    #plot_train_val_data(sub, data, "r2_mean")

#r2 map mean

# f = plt.figure()
# ax = plt.subplot(1,1,1)
# plot_stat_map(a, figure=f)
# f.savefig(os.path.join(out_directory, key+'.jpg'))
# plt.close
