import os
import warnings
from itertools import islice
from tqdm import tqdm
from random import shuffle, sample

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from nilearn.plotting import plot_stat_map
from nilearn.regions import signals_to_img_labels
from datetime import datetime
from matplotlib import pyplot as plt 

#1 - get input
from files_utils import fetchMRI
from audio_utils import convert_Audio

#1.1 - get films + subjects parcellation paths
stimuli_path = '/home/maelle/Database/cneuromod/movie10/stimuli' #'/home/brain/Data_Base/cneuromod/movie10/stimuli' 
stimuli_dic = {}
for film in os.listdir(stimuli_path):
    film_path = os.path.join(stimuli_path, film)
    if os.path.isdir(film_path):
        film_wav = [os.path.join(film_path, seg) for seg in os.listdir(film_path) if seg[-4:] == '.wav']
        stimuli_dic[film] = sorted(film_wav)

path_parcellation = '/home/maelle/Database/movie10_parc'#'/home/brain/Data_Base/movie10_parc'
all_subs = []
for sub_dir in sorted(os.listdir(path_parcellation)):
    sub_path = os.path.join(path_parcellation, sub_dir)
    all_subs.append([os.path.join(sub_path, mri_data) for mri_data in os.listdir(sub_path) if mri_data[-4:]==".npz"])

for i, sub in enumerate(all_subs) : 
    sub_segments = {}
    for film, segments in stimuli_dic.items() : 
        sub_segments[film] = []
        for j in range(len(segments)):
            sub_segments[film].extend(fetchMRI(segments[j], sub))

        all_subs[i] = sub_segments

#sub_parc[film] = [(seg1, run1),(seg2, run2),(seg3, run3), ...]

#2 - input processing
#3 - Create Dataset
from Datasets_utils import SequentialDataset
from audio_utils import load_audio_by_bit
import librosa

#load files !!!!!
bourne = 'bourne_supremacy'
wolf = 'wolf_of_wall_street'
life = "life"
hidden = "hidden_figures"
film = bourne
DataTest = all_subs[0][film]

tr=1.49
sr = 22050

print("getting audio files ...")
x = []
for (audio_path, mri_path) in DataTest :
    length = librosa.get_duration(filename = audio_path)
    audio_segment = load_audio_by_bit(audio_path, 0, length, bitSize = tr, sr = sr)
    x.append(audio_segment)
print("getting fMRI files ...")  
y = [np.load(mri_path)['X'] for (audio_path, mri_path) in DataTest]
print("done.")
#resize matrix to have the same number of tr : 

for i, (seg_wav, seg_fmri) in enumerate(zip(x, y)) : 
    min_len = min(len(seg_fmri), len(seg_wav))
    x[i] = seg_wav[:min_len]
    y[i] = seg_fmri[:min_len]

train_percent = 0.6
test_percent = 0.2
val_percent = 1 - train_percent - test_percent

dataset = SequentialDataset(x, y, batch_size=60)

total_len = len(dataset)
train_len = int(np.floor(train_percent*total_len))
val_len = int(np.floor(val_percent*total_len))
test_len = total_len-train_len-val_len
print(f'size of total, train, val and set : ', total_len, train_len, test_len, val_len)


loader = DataLoader(dataset, batch_size=None)
#trainloader = islice(loader, 0, train_len)
#valloader = islice(loader, train_len, train_len+val_len)
#testloader = islice(loader, train_len+val_len, None)

loader = list(loader)
trainloader = sample(loader[:train_len], k=train_len)
valloader = sample(loader[train_len:train_len+val_len], k=val_len)
testloader = sample(loader[train_len+val_len:train_len+val_len+test_len], k=test_len)

#4 - Models
from models import soundnet_model as sdn
from models import encoding_models as encod
#|--------------------------------------------------------------------------------------------------------------------------------------
### Model Setup
destdir = "/home/maelle/Results/encoding_11_2020/"+film
nroi=210
fmrihidden=1000
lr = 0.01
nbepoch = 100

net = encod.SoundNetEncoding_conv(pytorch_param_path='./sound8.pth',fmrihidden=fmrihidden,nroi_attention=nroi)
net.to("cpu")
mseloss = nn.MSELoss(reduction='sum')

### Optimizer and Schedulers
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,patience=5,threshold=1e-2,cooldown=2)

enddate = datetime.now()

#---------------------------------------------------------------------------------------------------------------------------------
#5 - Train & Test

from train_utils import train, test, test_r2, EarlyStopping
early_stopping = EarlyStopping(patience=10, verbose=True,delta=1e-6)

### Main Training Loop 
startdate = datetime.now()

train_loss = []
train_r2_max = []
train_r2_mean = []
val_loss = []
val_r2_max = []
val_r2_mean = []
try:
    for epoch in tqdm(range(nbepoch)):
        t_l, t_r2 = train(epoch,trainloader,net,optimizer,mseloss=mseloss)
        train_loss.append(t_l)
        train_r2_max.append(max(t_r2))
        train_r2_mean.append(np.mean(t_r2))

        v_l, v_r2 = test(epoch,valloader,net,optimizer,mseloss=mseloss)
        val_loss.append(v_l)
        val_r2_max.append(max(v_r2))
        val_r2_mean.append(np.mean(v_r2))
        print("Train Loss {} Train Mean R2 :  {} Train Max R2 : {}, Val Loss {} Val Mean R2:  {} Val Max R2 : {} ".format(train_loss[-1],train_r2_mean[-1],train_r2_max[-1],val_loss[-1],val_r2_mean[-1],val_r2_max[-1]))
        lr_sched.step(val_loss[-1])

        # early_stopping needs the R2 mean to check if it has increased, 
        # and if it has, it will make a checkpoint of the current model
        r2_forEL = -(val_r2_max[-1])
        early_stopping(r2_forEL, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

except KeyboardInterrupt:
    print("Interrupted by user")

test_loss = test(1,testloader,net,optimizer,mseloss=mseloss)
print("Test Loss : {}".format(test_loss))

#6 - Visualisation

mistroifile = '/home/maelle/Database/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'

dt_string = enddate.strftime("%Y-%m-%d-%H-%M-%S")
str_bestmodel = os.path.join(destdir,"{}.pt".format(dt_string))
str_bestmodel_plot = os.path.join(destdir,"{}.png".format(dt_string))
str_bestmodel_nii = os.path.join(destdir,"{}.nii.gz".format(dt_string))

r2model = test_r2(testloader,net,mseloss)
r2model[r2model<0] = 0
print("mean R2 score on test set  : {}".format(r2model.mean()))

print("max R2 score on test set  : {}".format(r2model.max()))

print("Training time : {}".format(enddate - startdate))

## Prepare data structure for checkpoint
state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'train_loss' : train_loss,
            'train_r2_max' : train_r2_max,
            'train_r2_mean' : train_r2_mean,
            'val_loss' : val_loss,
            'val_r2_max' : val_r2_max,
            'val_r2_mean' : val_r2_mean,
            'test_loss' : test_loss,
            'r2' : r2model,
            'r2max' : r2model.max(),
            'r2mean' : r2model.mean(),
            'training_time' : enddate - startdate,
            'nhidden' : fmrihidden,
            'model' : net
        }


### Plot the loss figure
f = plt.figure(figsize=(20,40))

ax = plt.subplot(4,1,2)

plt.plot(state['train_loss'][1:])
plt.plot(state['val_loss'][1:])
plt.legend(['Train','Val'])
plt.title("loss evolution => Mean test R^2=${}, Max test R^2={}, for model {}, batchsize ={} and {} hidden neurons".format(r2model.mean(),r2model.max(), "sdn_1_conv", str(60), fmrihidden))

### Mean R2 evolution during training
ax = plt.subplot(4,1,3)

plt.plot(state['train_r2_mean'][1:])
plt.plot(state['val_r2_mean'][1:])
plt.legend(['Train','Val'])
plt.title("Mean R^2 evolution for model {}, batchsize ={} and {} hidden neurons".format("sdn_1_conv", str(60), fmrihidden))

### Max R2 evolution during training
ax = plt.subplot(4,1,4)

plt.plot(state['train_r2_max'][1:])
plt.plot(state['val_r2_max'][1:])
plt.legend(['Train','Val'])
plt.title("Max R^2 evolution for model {}, batchsize ={} and {} hidden neurons".format("sdn_1_conv", str(60), fmrihidden))

### R2 figure 
r2_img = signals_to_img_labels(r2model.reshape(1,-1),mistroifile)

ax = plt.subplot(4,1,1)

plot_stat_map(r2_img,display_mode='z',cut_coords=8,figure=f,axes=ax)
f.savefig(str_bestmodel_plot)
r2_img.to_filename(str_bestmodel_nii)
plt.close()

torch.save(state, str_bestmodel)

#7 - User Interface