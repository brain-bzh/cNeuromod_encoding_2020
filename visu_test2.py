from matplotlib import pyplot as plt
import files_utils as fu
import os
from torch import load, device
from nilearn import image, plotting, datasets, surface
from nilearn.input_data import NiftiMasker
import numpy as np

def total_append (array1, array2, axis=0):
    if array1.size == 0:
        array1 = np.array(array2)
    else:
        array1 = np.append(array1, array2, axis=axis)
    
    return array1


def one_train_plot_TEST(data, measure, colors = ['b', 'g', 'm', 'r']) : 
    legends = []
    for color, value in zip(colors, data):
        cdt = value['condition']

        train_data = value['train_'+str(measure)]['mean'][1:20]
        (train_min, train_max) = value['train_'+str(measure)]['error_bar']

        val_data = value['val_'+str(measure)]['mean'][1:20]
        (val_min, val_max) = value['val_'+str(measure)]['error_bar']

        plt.errorbar(range(len(train_data)), train_data, fmt=color+'-')
        plt.errorbar(range(len(val_data)), val_data, fmt=color+'--')
        
        legends.append(cdt+'_Train')
        legends.append(cdt+'_Val')
    plt.legend(legends, loc='upper left', bbox_to_anchor=(1,1))

if __name__ == "__main__":
    datapath = '/home/maelle/Results/20210309_optim_MIST_ROI210_lr0.01_100epochs_embed2020norm/subject_0'
    targets = ['wd']
    all_ks = [1,5,10]
    all_wd = ['0','0.1','0.01','0.001']
    out_directory = '/home/maelle/Results/analysis_212103_Nicolas/gros_test/'
    prefix = '2020_MIST_ROI210_lr10e-3_sub0_100epochs'
    fu.create_dir_if_needed(out_directory)

    measures = ["loss", "r2_max", "r2_mean"]
    step_train = ["train", "val", "test"]

    network_keys = []
    for measure in measures:
        for step in step_train:
            network_keys.append(step+'_'+measure)

    for ks in all_ks:
        #une figure par ks, tout film confondu
        conditions = {'null':[], 'decoupled':[] , 'pt': [], 'dec_pt' : []}


        for film in os.listdir(datapath):
            film_path = os.path.join(datapath, film)
            for data_file in os.listdir(film_path):
                file_path = os.path.join(film_path, data_file)
                basename, ext = os.path.splitext(data_file)
                ks_value = fu.extract_value_from_string(basename, basename.find('ks')+len('ks'))
                if ext != '' and ks_value == ks: 
                    wd_idx = basename.find('wd')
                    wd_value = fu.extract_value_from_string(basename, wd_idx+len('wd'))
                    if ext == '.pt':

                        data = load(file_path, map_location=device('cpu'))
                        #select results only : train_data, val_data, ...
                        new_data = {}
                        for key in network_keys:
                            new_data[key] = data[key][0] if key == 'test_loss' else data[key] 
                        data = (wd_value, film, ks_value, new_data)
                        if basename.find('decoupled')>-1 and basename.find('pt')>-1:
                            conditions['dec_pt'].append(data)
                        elif basename.find('decoupled')>-1 :
                            conditions['decoupled'].append(data)
                        elif basename.find('pt')>-1 : 
                            conditions['pt'].append(data)
                        else : 
                            conditions['null'].append(data)

        all_wd = {'0':[], '10-4':[], '10-3':[], '10-2':[]}
        for condition, value in conditions.items():
            value.sort()
            for i, keyA in enumerate(all_wd.keys()):
                start_index, end_index = 4*i, 4*(i+1)
                all_films = value[start_index:end_index]
                #print(all_films[0][3]['test_r2_max'])
                all_films_data = {'ks_value':all_films[0][2], 'wd':all_films[0][0], 'condition':condition}
                for key in network_keys:
                    raw_data = np.array([all_films[j][3][key] for j in range(4)])
                    mean_data = np.mean(raw_data, axis=0)
                    max_data = np.max(raw_data, axis=0)
                    min_data = np.min(raw_data, axis=0)
                    all_films_data[key] = {'mean':mean_data, 'error_bar' : (min_data, max_data)}
                all_wd[keyA].append(all_films_data)

        f = plt.figure(figsize=(5*len(measures)+7, 7*len(all_wd)))

        for i, (wd_value, wd_data) in enumerate(all_wd.items()):
            for j, measure in enumerate(measures):
                ax = plt.subplot(len(all_wd),len(measures),len(measures)*i+(j+1))
                one_train_plot_TEST(wd_data, measure)
                plt.title('mean '+str(measure)+
                ' in all films for wd : '+str(wd_value)
                +', test_'+str(measure)+' : '+str(wd_data[i]['test_'+measure]['mean']))

        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        f.savefig(os.path.join(out_directory, prefix+'ks_{}_visu20'.format(ks)))
        plt.close()
