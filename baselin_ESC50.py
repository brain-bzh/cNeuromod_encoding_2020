 #Pour esc50, pourrais tu préparer la baseline, qui consiste en extraire les vecteurs avec SoundNet original sans finetuning, 
 #puis faire la classification ? Je voudrais relire ça pour te faire un retour
import os
import tqdm
import numpy as np
import pandas as pd
import librosa
import torch
import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize, scale
from scipy.signal import resample

from models import soundnet_model as snd

esc_datapath = "/home/maelle/DataBase/ESC-50-master/audio"
pytorch_param_path = './sound8.pth'

df_esc50 = pd.read_csv('/home/maelle/DataBase/ESC-50-master/meta/esc50.csv')
df_esc10 = df_esc50[df_esc50["esc10"] == True]
dataset = df_esc10

offset = 0 #starting point for audio
#The data is prearranged into 5 folds and the accuracy results are reported as the mean of 5 leave-one-fold-out evaluations.

#à checker
#sklearn preprocessing scale or normalize ?
#label majoritaire -> 3 steps : get all predict for 1 second + real_y, label majo, accuracy

soundnet = snd.SoundNet8_pytorch()
soundnet.load_state_dict(torch.load(pytorch_param_path))
output_layer = 'conv6' #output de pool5
overlap = 5

#parameter_grid = {'C':[1e-2, 1e-1, 1, 10]}

#------category_labels--------------------------------------------------------------
target_labels = pd.DataFrame([dataset['target'], dataset['category']]).T.drop_duplicates().sort_values(by=['target']).reset_index(drop=True)
print(target_labels)

#----------folds-WIP----------------------------------------------------

folds = {1:[], 2:[], 3:[], 4:[], 5:[]}
train_scores = []
test_scores = []

#---------------creation of X-Y pairs---------------------------------------
for filename, target, fold in tqdm.tqdm(zip(dataset['filename'], dataset['target'], dataset['fold'])) : 
    filepath = os.path.join(esc_datapath, filename)
    audio_x, sr = librosa.core.load(filepath, sr=None, offset = offset)
    length = librosa.get_duration(audio_x,sr)
    x = resample(audio_x, int(length)*22050)
    
    #------------data augmentation---------------------------------------------
    total_tp = len(x)
    dur_sample = 22050 #nb de sample pour chaque exemple
    step = int(dur_sample/overlap)

    for i, start in enumerate(range(0, total_tp, step)) : 
        sub_x = x[start:start+dur_sample]
        if len(sub_x) == dur_sample : 
            sub_x = torch.tensor(sub_x).view(1,1,-1, 1)
            sub_x = soundnet(sub_x, output_layer)
            sub_x = torch.squeeze(sub_x).T
            sub_x = sub_x.reshape(-1)
            #x = torch.mean(sub_x,0)
            folds[fold].append((sub_x, target))
        else : 
            pass

#--------------------separation train/test -------------------------------------

for test_fold in range(1,6):
    folds_temp = folds.copy()
    print(f'test fold : ', test_fold)
    datatest = folds_temp.pop(test_fold)
    datatrain = []
    for num, fold in folds_temp.items() : 
        datatrain.extend(fold)
    print(f'length datatest : ', len(datatest), ', length datatrain : ', len(datatrain))

    x_train = normalize(np.array([x.detach().numpy() for x,y in datatrain]))
    y_train = np.array([y for x,y in datatrain])

    x_test = normalize(np.array([x.detach().numpy() for x,y in datatest]))
    y_test = np.array([y for x,y in datatest])

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    #------------------------------train SVM------------------------------------------

    net = svm.LinearSVC(dual=False, C=1e-2)
    #clf = GridSearchCV(net, parameter_grid)
    net.fit(x_train, y_train)
    #print(clf.cv_results_['rank_test_score'])
    train_scores.append(net.score(x_train,y_train))
    test_scores.append(net.score(x_test,y_test))

train_score = np.mean(train_scores)
test_score = np.mean(test_scores)
print(train_scores)
print(test_scores)
print(f'train score : ', train_score, ', test score : ', test_score)

    





