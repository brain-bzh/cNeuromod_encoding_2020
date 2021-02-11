from matplotlib import pyplot as plt
import files_utils as fu
import os
from torch import load, device

def one_train_plot_TEST(criterion, data, measure, colors = ['b', 'g', 'm', 'r']) : 
    legends = []

    for color, test in zip(colors, data):
        key = test[0]
        data_dict = test[1]
        plt.plot(data_dict['train_'+str(measure)], color+'-')
        plt.plot(data_dict['val_'+str(measure)], color+'--')
        #plt.text(14, 1, 'test_'+str(measure)+' : '+str(data_dict['test_'+str(measure)]))
        legends.append(key+'_Train')
        legends.append(key+'_Val')

    plt.legend(legends, loc='upper right')
    plt.title(str(measure)+' in '+str(criterion))

if __name__ == "__main__":
    datapath = '/home/maelle/Results/20210211_tests_kernel_MIST_ROI_embed_2020_norm/subject_0/bourne_supremacy'
    target = 'ks'
    out_directory = os.path.join(datapath, 'analysis')
    fu.create_dir_if_needed(out_directory)

    all_data = []
    for data_file in os.listdir(datapath):
        file_path = os.path.join(datapath, data_file)
        basename, ext = os.path.splitext(file_path)
        if ext == '.pt':
            tgt_idx = basename.find(target)
            tgt_value = fu.extract_value_from_string(basename, tgt_idx)
            data = load(file_path, map_location=device('cpu'))
            all_data.append((target+'_'+str(tgt_value), data))


    measures = ["loss", "r2_max", "r2_mean"]
    for measure in measures:
        f = plt.figure()
        one_train_plot_TEST('bourne', all_data, measure)
        f.savefig(os.path.join(out_directory, 'bourne_{}'.format(target)))


        # for sub, films in all_data.items():
        # all_loaded = [(dir_name, load(file_path, map_location=device('cpu')), target_data) for (dir_name, file_path, target_data) in films]
        # all_data[sub] = all_loaded
