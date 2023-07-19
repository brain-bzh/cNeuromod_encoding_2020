from datetime import datetime
import os
from re import sub
from librosa.core import audio
from torch.functional import norm
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import torch
import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, scale,StandardScaler
from sklearn.metrics import classification_report
from scipy.signal import resample
#from umap import UMAP
from matplotlib import pyplot as plt
from models import soundnet_model as snd
import argparse


parser = argparse.ArgumentParser(epilog='With all default parameters, you should obtain average scores accross folds for train:  0.9914583333333333 , and test :  0.8125',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--esc", type=str,default="/home/maelle/DataBase/ESC-50-master/",help="Path to the ESC-50 folder (must be cloned from https://github.com/karolpiczak/ESC-50)")
parser.add_argument("--checkpoint","-c", type=str,default='./sound8.pth',help="Path to soundnet checkpoint to load, default is the trained soundnet from the paper")
parser.add_argument("--dataset", type=str,default='esc10',choices=['esc10','esc50'],help="esc10 or esc50")
parser.add_argument("--layer", type=str,default='conv6',choices=['conv1', 'pool1', 'conv2', 'pool2' , 'conv3' , 'conv4' , 
                            'conv5', 'pool5', 'conv6', 'conv7'],help="Choose soundnet layer to train from")
parser.add_argument("--feat", type=str,default='mean',choices=['mean', 'concat'],help="Choose whether to average or concatenate the features")                            
parser.add_argument("--seg", type=float,default='3.0',help="Segment length (s) to divide the wav files ")
parser.add_argument("--step", type=float,default='1.0',help="Step length (s) between segments")
parser.add_argument("--padding", type=float,default='5.0',help="How much padding (s) to add before and after the segments")
parser.add_argument("--classif", type=str,default='mlp',choices=['svm', 'mlp', 'knn'],help="Choose classifer (SVM, MLP, KNN)")
parser.add_argument("--hidden", type=int,default=512,help="Number of hidden neurons")
parser.add_argument("--save", type=str,default="results.csv",help="Path to a csv file to aggregate results (must have been saved with this script, will be loaded if exists or created otherwise)")

args = parser.parse_args()
### See here https://github.com/nicofarr/paper-2015-esc-dataset/tree/master/Notebook for baseline results using MFCC extracted features 
### including results and code (notebook with lots of graphs / details)

esc_datapath = args.esc + '/audio/' #"/home/nfarrugi/git/ESC-50/audio/"
pytorch_param_path = args.checkpoint

df_esc50 = pd.read_csv(os.path.join(args.esc , 'meta/esc50.csv'))
df_esc10 = df_esc50[df_esc50["esc10"] == True]

if args.dataset == 'esc10':
    dataset = df_esc10
    print("Training on ESC10")
else:
    dataset = df_esc50
    print("Training on ESC50")

offset = 0 #starting point for audio
#The data is prearranged into 5 folds and the accuracy results are reported as the mean of 5 leave-one-fold-out evaluations.

soundnet = snd.SoundNet8_pytorch()
soundnet.load_state_dict(torch.load(pytorch_param_path))
output_layer = args.layer #by default, conv6 is output de pool5 (--> to check)
sr_soundnet = 22050

parameter_grid = {'C':[1e-2,1e-1,1]}

#------category_labels--------------------------------------------------------------
target_labels = pd.DataFrame([dataset['target'], dataset['category']]).T.drop_duplicates().sort_values(by=['target']).reset_index(drop=True)
#print(target_labels)

#----------folds-WIP----------------------------------------------------

folds = {1:[], 2:[], 3:[], 4:[], 5:[]}
train_scores = []
test_scores = []

#---------------creation of X-Y pairs---------------------------------------
for filename, target, fold in tqdm(zip(dataset['filename'], dataset['category'], dataset['fold']),total=len(dataset['filename'])) : 
    filepath = os.path.join(esc_datapath, filename)
    audio_x, sr = librosa.core.load(filepath, sr=None, offset = offset)
    length = librosa.get_duration(audio_x,sr)
    x = 256*resample(audio_x, int(length)*sr_soundnet) ### Range in the SoundNet paper is [-256,256]
    
    """
    sub_x = torch.tensor(x).view(1,1,-1, 1)
    sub_x = soundnet(sub_x, output_layer)
    sub_x = torch.squeeze(sub_x).T
    #sub_x = sub_x.reshape(-1)
    sub_x = torch.mean(sub_x,0)
    folds[fold].append((sub_x, target,mfccs_scaled)) 
    
    #### BASELINE ESC-10
    ##### Without DA and without padding : SVM Grid Search 1e-2,1e-1, conv6 (out pool5), mean of feature vec : train score :  0.9956250000000001 , test score :  0.7474999999999999
"""
    
    
    #------------data augmentation (overlapping shorter segments)---------------------------------------------
    ### Setting dur_sample to 5 seconds disables DA

    total_tp = len(x)
    dur_sample = int(sr_soundnet*args.seg)#nb de sample pour chaque example
    
    step = int(sr_soundnet*args.step)

    for i, start in enumerate(range(0, total_tp, step)) : 
        sub_x = x[start:start+dur_sample]
        
        if len(sub_x) == dur_sample : 
            ### padding by copying the first and final sample many times at the beginning and end
            pad_length = int(args.padding * sr_soundnet) ### for example 8 * sr_soundnet will add 8 seconds at the beginning and end  
            pad_beg = np.ones_like(sub_x)[:pad_length]
            pad_end = np.ones_like(sub_x)[:pad_length]
            sub_x = np.hstack([pad_beg,sub_x,pad_end])
            #print(pad_beg.shape,sub_x.shape,pad_end.shape)
            
            ## Calculate SoundNet features

            sub_x = torch.tensor(sub_x).view(1,1,-1, 1)
            sub_x = soundnet(sub_x, output_layer)
            sub_x = torch.squeeze(sub_x).T

            if args.feat == 'concat':
                ###  try to keep only the non padded part, but it's a little tricky to calculate... 
                ###                 
                ratio = pad_length / dur_sample  
                cut = int((sub_x.shape[0]/ratio)/2)
                
                sub_x = sub_x[cut:-cut,:]
                
                sub_x = sub_x.reshape(-1).detach().numpy() #### better to detach here to save memory   
                
            else:
                sub_x = torch.mean(sub_x,0).detach().numpy() #### better to detach here to save memory
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

    x_train = StandardScaler().fit_transform((np.array([x for x,y in datatrain]))) ### Standardize features by removing the mean and scaling to unit variance
    y_train = np.array([y for x,y in datatrain])


    x_test = StandardScaler().fit_transform(np.array([x for x,y in datatest])) ### Standardize features by removing the mean and scaling to unit variance
    y_test = np.array([y for x,y in datatest])

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    #------------------------------train SVM------------------------------------------
    ## A few very simple estimators 

    #clf = svm.LinearSVC(C=1e-1)
    if args.classif == 'svm':
        print("Grid Search with Linear SVM")
        clf = GridSearchCV(svm.LinearSVC(dual=False), parameter_grid,n_jobs=-1)
    elif args.classif == 'mlp':
        print('Classifying with MLP')
        clf = MLPClassifier(hidden_layer_sizes=(args.hidden,),learning_rate_init=0.01)
    else:
        print('Classifying with KNN')
        clf=KNeighborsClassifier()

    clf.fit(x_train, y_train)
    #print(clf.cv_results_['rank_test_score'])
    train_scores.append(clf.score(x_train,y_train))
    test_scores.append(clf.score(x_test,y_test))
    

    print('TRAINING SET')
    print(classification_report(y_true=y_train,y_pred=clf.predict(x_train)))

    print('TEST SET')
    print(classification_report(y_true=y_test,y_pred=clf.predict(x_test)))


    ### UMAP visualisation 

    #lowdim = UMAP(n_components=2).fit_transform(x_train)
    #plt.scatter(lowdim[:,0],lowdim[:,1],c=(y_train))
    #plt.show()


train_score = np.mean(train_scores)
test_score = np.mean(test_scores)
print(train_scores)
print(test_scores)
print(f'Average score accross folds for train: ', train_score, ', and test : ', test_score)
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
results = {'datetime':now,'dataset':args.dataset,'layer':args.layer,'feat':args.feat,'classifier':args.classif,'hidden':args.hidden,'seg':args.seg,'step':args.step,'padding':args.padding,'train_avg':train_score,'test_avg':test_score,'train_std': np.std(train_scores),'test_std':np.std(test_scores),
}

if os.path.isfile(args.save):
    Df = pd.read_csv(args.save)
    Df2 = pd.DataFrame([results])
    Df = pd.concat([Df,Df2])
else:
    Df = pd.DataFrame([results])

print(Df)
Df = Df.sort_values(by='test_avg',ascending=False)
Df.to_csv(args.save,index=False)