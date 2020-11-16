import os
import warnings
from itertools import islice
from tqdm import tqdm
from random import sample,shuffle
import librosa

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from train_utils import train, test, test_r2

from sklearn.metrics import r2_score

from nilearn.plotting import plot_stat_map
from nilearn.regions import signals_to_img_labels

#1 - get input

def fetchMRI(videofile,fmrilist):
    ### isolate the mkv file (->filename) and the rest of the path (->videopath)

    videopath,filename = os.path.split(videofile)
    #formatting the name to correspond to mri run formatting
    name = filename.replace('_', '')
    if name.startswith('the'):
        name = name.replace('the', '', 1)
    if name.find('life') > -1 :
        name = name.replace('life1', 'life')

    name = name.replace('seg','_run-')
    name = name.replace('subsampl','')
    ## Rename to match the parcellated filenames
    name = name.replace('.wav','npz.npz')

    # list of all parcellated filenames 

    # match videofilename with parcellated files
    mriMatchs = []
    for curfile in fmrilist:
        _, cur_name = os.path.split(curfile)
        if cur_name[23:] == (name):
            mriMatchs.append(curfile)    
    #in case of multiple run for 1 film segment
    name_seg = filename[:-4]

    if len(mriMatchs) > 1 :
        numSessions = []
        for run in mriMatchs :
            index_sess = run.find('ses-vid')
            numSessions.append(int(run[index_sess+7:index_sess+10]))
            
        if numSessions[0] < numSessions[1] : 
            return [(videofile, mriMatchs[0]), (videofile, mriMatchs[1])]

        else : 
            return [(videofile, mriMatchs[1]), (videofile, mriMatchs[0])]
    else :
        return [(videofile, mriMatchs[0])]

#1.1 - get films + subjects parcellation paths
stimuli_path = '/home/maelle/Database/cneuromod/movie10/stimuli'
stimuli_dic = {}
for film in os.listdir(stimuli_path):
    film_path = os.path.join(stimuli_path, film)
    if os.path.isdir(film_path):
        film_wav = [os.path.join(film_path, seg) for seg in os.listdir(film_path) if seg[-4:] == '.wav']
        stimuli_dic[film] = sorted(film_wav)

path_parcellation = '/home/maelle/Database/movie10_parc'
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
class SequentialDataset(IterableDataset):
    def __init__(self, x, y, batch_size):
        super(SequentialDataset).__init__()

        self.x = x
        self.y = y
        self.batch_size = batch_size

        self.batches = []
        for i, (seg_x, seg_y) in enumerate(zip(x,y)):
            seg = self.__create_batchs__(seg_x, seg_y)
            self.batches.extend(seg)
        self.batches = sample(self.batches, len(self.batches))

    def __create_batchs__(self, dataset_x, dataset_y):
        batches = []
        total_nb_inputs = len(dataset_x)
        for idx, batch_start in enumerate(range(0, total_nb_inputs, self.batch_size)):
            if batch_start+self.batch_size>total_nb_inputs:
                batch_end = total_nb_inputs
            else:
                batch_end = batch_start+self.batch_size
            batches.extend((idx, dataset_x[batch_start:batch_end], dataset_y[batch_start:batch_end]))    
            batches = sample(batches, len(batches))
        return batches

    def __len__(self):
        return(len(self.batches)) 

    def __iter__(self):
        return iter(self.batches)

#load files !!!!!
bourne = 'bourne_supremacy'
wolf = 'wolf_of_wall_street'
life = "life"
hidden = "hidden_figures"
film = bourne
DataTest = all_subs[0][film]

tr=1.49
sr = 22050

x = []
for (audio_path, mri_path) in DataTest :
    length = librosa.get_duration(filename = audio_path)
    audio_segment = []
    for start in np.arange(0, length, tr) : 
        (audio_chunk, _) = librosa.core.load(audio_path, sr=sr, mono=True, offset = start, duration = tr)
        audio_segment.append(audio_chunk)
    x.append(audio_segment)
    
y = [np.load(mri_path)['X'] for (audio_path, mri_path) in DataTest]

#resize matrix to have the same number of tr : 

for i, (seg_wav, seg_fmri) in enumerate(zip(x, y)) : 
    min_len = min(len(seg_fmri), len(seg_wav))
    x[i] = seg_wav[:min_len]
    y[i] = seg_fmri[:min_len]


#xDim = (len(x), len(x[0]), len(x[0][0]))
#yDim = (len(y), len(y[0]), len(y[0][0]))
#function to define dimensions of a list --> to convert into numpy array for a more efficient code
#do and adapt after.

train_percent = 0.6
test_percent = 0.2
val_percent = 1 - train_percent - test_percent

dataset = SequentialDataset(x, y, batch_size=60)

total_len = len(dataset)
train_len = int(np.floor(train_percent*total_len))
val_len = int(np.floor(val_percent*total_len))
test_len = total_len-train_len-val_len
print(total_len, train_len, test_len, val_len)


loader = DataLoader(dataset, batch_size=None)
trainloader = islice(loader, 0, train_len)
valloader = islice(loader, train_len, train_len+val_len)
testloader = islice(loader, train_len+val_len, None)

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
net = net.cuda()
mseloss = nn.MSELoss(reduction='sum')

### Optimizer and Schedulers
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,patience=5,threshold=1e-2,cooldown=2)

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
#print("Test Loss : {}".format(test_loss))

enddate = datetime.now()

#---------------------------------------------------------------------------------------------------------------------------------
#5 - Train & Test


#6 - Visualisation

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