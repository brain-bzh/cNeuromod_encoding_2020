import os
import numpy as np
import pickle
from torch import load

from nilearn import image, plotting
from nilearn.plotting import plot_stat_map
from nilearn.regions import signals_to_img_labels

from matplotlib import pyplot as plt 

out_directory = '/home/maelle/Results/encoding_11_2020/analysis'

bourne_map = '/home/maelle/Results/encoding_11_2020/bourne_supremacy/2020-12-07-18-12-36.nii.gz'
bourne_data = '/home/maelle/Results/encoding_11_2020/bourne_supremacy/2020-12-07-18-12-36.pt'

wolf_map = '/home/maelle/Results/encoding_11_2020/wolf_of_wall_street/2020-12-07-20-19-21.nii.gz'
wolf_data = '/home/maelle/Results/encoding_11_2020/wolf_of_wall_street/2020-12-07-20-19-21.pt'

hidden_map = '/home/maelle/Results/encoding_11_2020/hidden_figures/2020-12-07-19-18-53.nii.gz'
hidden_data = '/home/maelle/Results/encoding_11_2020/hidden_figures/2020-12-07-19-18-53.pt'

life_map = '/home/maelle/Results/encoding_11_2020/life/2020-12-07-20-06-15.nii.gz'
life_data = '/home/maelle/Results/encoding_11_2020/life/2020-12-07-20-06-15.pt'

all_data = {'bourne':bourne_data, 'wolf':wolf_data, 'hidden':hidden_data, 'life':life_data}
all_maps = {'bourne':bourne_map, 'wolf':wolf_map, 'hidden':hidden_map, 'life':life_map}
#(1) superposer les courbes d'apprentissage pour les différents films, et  
# (2) faire la moyenne des cartes de R2 entre films?

for key, value in all_data.items():
    all_data[key] = load(value)

for key, value in all_maps.items() :
    all_maps[key] = image.load_img(value)

colors = ['b', 'g', 'm', 'r']

#loss
loss = plt.figure()
legends = []
for color, (key, data_dict) in zip(colors,all_data.items()):
    plt.plot(data_dict['train_loss'], color+'-')
    plt.plot(data_dict['val_loss'], color+'--')
    legends.append(key+'_Train')
    legends.append(key+'_Val')

plt.legend(legends, loc='upper right')
plt.title("loss")
loss.savefig(os.path.join(out_directory, 'all_films_loss.jpg'))
plt.close()

#r2_max
r2_max = plt.figure()
legends = []
for color, (key, data_dict) in zip(colors,all_data.items()):
    plt.plot(data_dict['train_r2_max'], color+'-')
    plt.plot(data_dict['val_r2_max'], color+'--')
    legends.append(key+'_Train')
    legends.append(key+'_Val')

plt.legend(legends, loc='upper right')
plt.title("r2_max")
r2_max.savefig(os.path.join(out_directory, 'all_films_r2_max.jpg'))
plt.close()


#r2_mean
r2_mean = plt.figure()
legends = []
for color, (key, data_dict) in zip(colors,all_data.items()):
    plt.plot(data_dict['train_r2_mean'], color+'-')
    plt.plot(data_dict['val_r2_mean'], color+'--')
    legends.append(key+'_Train')
    legends.append(key+'_Val')

plt.legend(legends, loc='upper right')
plt.title("r2_mean")
r2_mean.savefig(os.path.join(out_directory, 'all_films_r2_mean.jpg'))
plt.close()


    
# f = plt.figure()
# ax = plt.subplot(1,1,1)
# plot_stat_map(a, figure=f)
# f.savefig(os.path.join(out_directory, key+'.jpg'))
# plt.close
