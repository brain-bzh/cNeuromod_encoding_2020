import os, argparse
import numpy as np
import pandas as pd
from torch import load, device
from nilearn import plotting
from matplotlib import pyplot as plt 
from scipy import corrcoef
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from visu_utils import ROI_map

import statistics

parser = argparse.ArgumentParser()
parser.add_argument("-n","--nbBestConfigs", type=int)
parser.add_argument("-c","--nClusters", type=int)
args = parser.parse_args()
nb_selected_config = args.nbBestConfigs
nClusters = args.nClusters
#pour MIST_ROI
#similarité par distribution spatiale :
#matrice 210*1000 fv (régions col / lignes run)
ROI_info_file = '/home/maelle/DataBase/fMRI_parcellations/MIST_parcellation/Parcel_Information/MIST_ROI.csv'
ROI_info = pd.read_csv(ROI_info_file, sep=';')
runs = np.load('./MIST_ROI_ordered_data_from_HPtrain_2021.npy')    
configs_df = pd.read_csv('./MIST_ROI_ordered_configs_from_HPtrain_2021', sep=';')
sorted_df = configs_df.sort_values(by=['val r2 max'], ascending=False)

sorted_df.rename( columns={'Unnamed: 0':'index'}, inplace=True )
sorted_df.drop(labels = 'Unnamed: 0.1', axis=1, inplace=True)
df_indexes = sorted_df['index']

indexes_list = [run[1] for run in runs]
sorted_df.set_index('index', inplace=True)

for _, row in df_indexes.iteritems():
    if row not in indexes_list:
        sorted_df.drop(labels = row, axis=0, inplace=True)

print(runs.shape, sorted_df.shape)
    #5 - création de la matrice ROI*runs:
runs2D = runs[:nb_selected_config]
restrict_df = sorted_df[:nb_selected_config]
blank_runs = []
for i, run in enumerate(runs2D):
    if np.max(run[2:]) == 0 and np.mean(run[2:]) == 0 :
        blank_runs.append(i)
        restrict_df.drop(index=run[1], axis=0, inplace=True)
runs2D = np.delete(runs2D, blank_runs, 0)
runs_no_index = np.delete(runs2D, [0,1], axis=1)
print(runs_no_index.shape, restrict_df.shape)

    #5.1 - matrice de corrélation des runs
R = corrcoef(runs_no_index)
plt.matshow(np.abs(R))
#plt.colorbar()
#plt.savefig('/home/maelle/Results/20220126_Hypertraining_analysis/TESTcorrMatrix{}_NoBlank.png'.format(nb_selected_config))
#plt.close()

    #6 - clustering hiérarchique (avec centroïdes ?) cf https://github.com/SIMEXP/tutorials-basc/blob/master/tutorial_basc_principles.ipynb
hier = linkage(runs_no_index, method='centroid', metric='euclidean')
res = dendrogram(hier, get_leaves=True)
#plt.savefig('/home/maelle/Results/20220126_Hypertraining_analysis/RoiRunTree{}.png'.format(nb_selected_config))
#plt.close()
#order = res.get('leaves')

    #7 - division en n clusters et calcul centroïde par cluster
part = cut_tree(hier, n_clusters=nClusters)
cluster_data = np.concatenate((part, runs2D), axis=1)
roiLabels = np.array(ROI_info['label'])
column_labels = np.concatenate((['cluster', 'rank', 'idx'], roiLabels))
df = pd.DataFrame(cluster_data, columns=column_labels)

for cluster in range(nClusters):
    cluster_hp_df = restrict_df.copy()
    cluster_df = df[df['cluster']==cluster]
    for index, row in cluster_hp_df.iterrows():
        if index not in list(cluster_df['idx']):
            cluster_hp_df.drop(index=index, axis=0, inplace=True)
    print(cluster_hp_df.head(5))    


    print('\nSize of cluster {} : {} configurations. \n'.format(cluster, cluster_df.shape[0]))
    hyperparameters = ['bs', 'lr', 'ks', 'wd', 'patience', 'delta']
    for hp in hyperparameters:
        a = cluster_hp_df[hp].value_counts(normalize=True)
        print(a)
        try : 
            best_hp = statistics.mode(cluster_hp_df[hp])
            print('dominant value of {} in cluster {} : {}'.format(hp, cluster, best_hp))

        except statistics.StatisticsError:
            print('no dominant value was found for {} in cluster {}'.format(hp, cluster))

#-------------------------------visu-cluster-------------------------------------------
# for cluster in range(nClusters):
#     cluster_df = df[df['cluster'] == cluster]
#     cluster_df = cluster_df.drop('cluster', axis=1)

#     nb_configs = cluster_df.shape[0]

#     mean_df = cluster_df.mean(axis=0)
#     max_r2 = mean_df.max()
#     mean_r2 = mean_df.mean()
    
#     max_df = cluster_df.max(axis=1)
#     max_max = max_df.max()
#     min_max = max_df.min()
    
#     moy_df = cluster_df.mean(axis=1)
#     max_mean = moy_df.max()
#     min_mean = moy_df.min()

    #-------------------------------------------------
    # x= [0, 1]
    # y= [max_r2, mean_r2]
    
    # # Define Error
    # y_error = [(min_max, max_max), (max_mean, min_mean)]
    # #Plot Bar chart
    # plt.bar(x,y)
    # # Plot error bar
    # plt.errorbar(x, y, yerr = y_error,fmt='o',ecolor = 'red',color='yellow')
    # # Display graph
    # plt.savefig('/home/maelle/Results/20220126_Hypertraining_analysis/r2_stat_for_cluster_{}_on_{}_from_the_best_{}_configurations'.format(cluster, nClusters, nb_selected_config))
    # plt.close()


    # #8 - map cluster
    # out_directory = '/home/maelle/Results/20220126_Hypertraining_analysis'
    # title = 'R2_map_cluster_{}_on_{}_from_the_best_{}_configurations'.format(cluster, nClusters, nb_selected_config)
    # data = np.squeeze(np.array(mean_df))
    # print(data.shape)
    # ROI_map(data, title, out_directory, threshold=0, display_mode='z')

#clustering/partitionnement de l'espace non supervisée - k-means, clustering hié#clusters = np.concatenate((part, hier), axis=1)rarchique (agglomerative, 7-10 ...) - coupe 3/4 ++
#centroïde (représentatif cluster --> point dans l'espace de départ --> possible à plotter)
#2 histogrammes r2max / r2mean superposés pour chaque cluster
#compte/proportion HP par cluster
#distribution de valeur r2 max/mean dans les clusters

