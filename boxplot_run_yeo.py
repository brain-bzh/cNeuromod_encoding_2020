import os
from nilearn import datasets, plotting
from torch import load, device

results_path = '/home/maelle/Results/converted_models'

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas_yeo = atlas_yeo_2011.colors_7
print('plouf')
print(atlas_yeo)




for path, dirs, files in os.walk(results_path):
    for f in files : 
        if '.pt' in f : 
            filepath = os.path.join(path, f)
            #data = load(filepath, map_location=device('cpu'))


# 7Networks_1 : purple            Visual
# 7Networks_2 : bleu vert mer     Somatomotor
# 7Networks_3 : vert forêt        dorsal attention
# 7Networks_4 : violet néon       ventral attention
# 7Networks_5 : vert lime pâle    limbic
# 7Networks_6 : orange foncé      FRontoparietal
# 7Networks_7 : rouge bordeaux    Default