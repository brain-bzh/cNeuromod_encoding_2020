import os
import warnings
from itertools import islice
from tqdm import tqdm
from random import sample,shuffle

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader

from sklearn.metrics import r2_score

from nilearn.plotting import plot_stat_map
from nilearn.regions import signals_to_img_labels

#first, I need to load the fmri data and frame_feature data
frame_features = np.load('frame_features.npy')
fmri_data = np.load('X.npy')

################ I - defining the model ##############################
nroi = 210
fmrihidden = 1000
batch_size = 10

train_percent = 0.6 
test_percent = 0.2
val_percent = 1 - train_percent - test_percent

# from the pytorch class nn.Module, I create a network model and I specify all the layers

class Encoding_model(nn.Module):
    def __init__(self, features_length, nroi=210,nb_hidden=1000):
        super(Encoding_model, self).__init__()

        self.features_length = features_length
        self.nb_hidden = nb_hidden
        self.nroi = nroi

        # Here, I define the sequence of tranformations that will change my input (with a specific feature length) 
        # into the ouput shape (number of roi in our case), through one hidden layer (with nb_hidden neurons)
        #the ReLU Layer is just a filter of weights, that will put to 0 every negative values.
        self.encoding_fmri = nn.Sequential(              
                nn.Conv1d(self.features_length,self.nb_hidden,kernel_size=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.nb_hidden,self.nroi,kernel_size=(1,1)),
            )

    #after instanciating the network ("my_network = my_model(arg1, arg2, ...)"), the forward function of network model is called 
    # when you write "predicted_output = my_network(input)"

    def forward(self, x):
        warnings.filterwarnings("ignore")
        return self.encoding_fmri(x)

net = Encoding_model(frame_features.shape[1], nroi=nroi, nb_hidden=fmrihidden)
net = net.cuda()

# Finally I define the different parameters needed for the training

nbepochs = 50
lr = 0.01
mseloss = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
#the lr_sched is scheduler that will adapt the learning rate during the training, so we can tend toward the more optimized result
lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,patience=5,threshold=1e-2,cooldown=2)
#early_stopping = EarlyStopping(patience=10, verbose=True,delta=1e-6)

############################### II - Construction of the dataset ###########################################""

#I need to create a class for my dataset, that will specify the structure of my dataset
   #just for testing
#alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#indices = [indice + 1 for indice in range(25)]

class SequentialDataset(IterableDataset):
    def __init__(self, x, y, batch_size):
        super(SequentialDataset).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        if isinstance(self.x, list) : 
            self.total_nb_inputs = len(self.x)
        else : 
            self.total_nb_inputs = x.shape[0]

        self.batches = self.__create_batchs__(self.x, self.y)

    def __create_batchs__(self, dataset_x, dataset_y):
        batches = []
        for idx, batch_start in enumerate(range(0, self.total_nb_inputs, self.batch_size)):
            if batch_start+self.batch_size>self.total_nb_inputs:
                batch_end = self.total_nb_inputs
            else:
                batch_end = batch_start+self.batch_size
            batches.append((idx, dataset_x[batch_start:batch_end], dataset_y[batch_start:batch_end]))    
        batches = sample(batches, len(batches))
        return batches

    def __len__(self):
        return(len(self.batches)) 

    def __iter__(self):
        return iter(self.batches)

#test
#dataset = sequential_dataset(alphabet, indices, batch_size=7)
#dataloader = data.DataLoader(dataset, batch_size=None)

dataset = SequentialDataset(frame_features, fmri_data, batch_size=10)
print(type(dataset))

#Now I create the dataloader, who will shuffle the batches, and I create one for validation, 
# one for training and one for test. The dataloader give the data to the training loop, 
# without modifying the original data

total_len = len(dataset)
train_len = int(np.floor(train_percent*total_len))
test_len = int(np.floor(val_percent*total_len))
val_len = int(np.floor(test_percent*total_len)) - 1

loader = DataLoader(dataset, batch_size=None)
trainloader = islice(loader, 0, train_len)
valloader = islice(loader, train_len, train_len+val_len)
testloader = islice(loader, train_len+val_len, None)

# for loader_test in [trainloader, valloader, testloader]:
#     print('\n new loader : ')
#     for b in islice(loader_test, 3) :
#         print(b)

##############################################################################################
#III - defining the training/test loop

def train(epoch,trainloader,net,optimizer,mseloss,delta=1e-2):
    all_y = []
    all_y_predicted = []
    running_loss = 0
    net.encoding_fmri.train()

    #looping on all the batches (batch_idx is here just to check if the batch are actually randomly picked)
    for batch_nb, (batch_idx, x, y) in enumerate(trainloader):
        optimizer.zero_grad()
        batch_size = x.shape[0]
        #print(batch_num, batch_idx, x.shape[0], y.shape[0])

        # for 1D output
        #print(f'x before becoming into a tensor and reshaping : ', type(x2), x2.shape)
        x2 =  torch.Tensor(x).view(batch_size,-1,1, 1)          #we need to adapt the shape of the batch so it can fit into pytorch convolutions layers
        #print(f'x before becoming into a tensor and reshaping : ', type(x2), x2.shape)
        x = x2.cuda()                                           #then we put the tensor into the memory of the graphic card, so computations can be done faster
        
        # Forward pass
        predicted_y = net(x)
        #print(f'predicted_y before squeezing : ', predicting_y)
        predicted_y = predicted_y.squeeze().double()                     # as some dimensions in the output 

        # predicted_y = predicted_y[:batch_size]                 #FOR AUDIO ONLY : Cropping the end of the predicted fmri to match the measured bold
        #print(f'predicted_y after squeezing and cutting to the same size as fmri data: ', predicting_y)

        y = y.double().cuda()
        #print(f"y_real shape : ", y.shape, "and y_predicted shape : ", predicted_y.shape)         # both must have the same shape
        loss=delta*mseloss(predicted_y,y)/batch_size
        loss.backward()
        optimizer.step()

        all_y.append(y.cpu().numpy().reshape(batch_size,-1))
        all_y_predicted.append(predicted_y.detach().cpu().numpy().reshape(batch_size,-1))
        running_loss += loss.item()
        

    r2_model = r2_score(np.vstack(all_y),np.vstack(all_y_predicted),multioutput='raw_values') 
    return running_loss/batch_nb, r2_model


def test(epoch,trainloader,net,optimizer,mseloss,delta=1e-2):
    all_y = []
    all_y_predicted = []
    running_loss = 0
    net.eval()

    with torch.no_grad() : 
        for batch_nb, (batch_idx, x, y) in enumerate(trainloader):
            optimizer.zero_grad()
            batch_size = x.shape[0]

            # for 1D output 
            x2 =  torch.Tensor(x).view(batch_size,-1,1, 1)
            x = x2.cuda()  
            # Forward pass
            predicted_y = net(x)
            predicted_y = predicted_y.squeeze().double()
            y = y.double().cuda()

            loss=delta*mseloss(predicted_y,y)/batch_size

            all_y.append(y.cpu().numpy().reshape(batch_size,-1))
            all_y_predicted.append(predicted_y.detach().cpu().numpy().reshape(batch_size,-1))
            running_loss += loss.item()
            

        r2_model = r2_score(np.vstack(all_y),np.vstack(all_y_predicted),multioutput='raw_values') 
        return running_loss/batch_nb, r2_model

#################################################################
#IV - defining the main loop, over all epochs + early stopping
### Main Training Loop 

def epoch_loop(net, nbepoch, trainloader, valloader, testloader, mseloss, optimizer):

    train_loss = []
    train_r2_max = []
    train_r2_mean = []
    val_loss = []
    val_r2_max = []
    val_r2_mean = []

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
        #r2_forEL = -(val_r2_max[-1])
        # early_stopping(r2_forEL, net)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    test_loss = test(1,testloader,net,optimizer,mseloss=mseloss)
    print("Test Loss : {}".format(test_loss))

############################################################
#start running the script

if __name__ == "__main__": 
    epoch_loop(net, nbepochs, trainloader, valloader, testloader, mseloss, optimizer)
    




#modified from Nicolas Farrugia
        #### Check if audio file exists
if os.path.isfile(self.wavfile) is False:

    #### If not, generate it and put it at the same place than the video file , as a wav, with the same name
    #### use this following audio file to generate predictions on sound 

    print('wav file does not exist, converting from {videofile}...'.format(videofile=videofile))

    convert_Audio(videofile, self.wavfile)

## Load just 2 seconds to check the sample rate 
wav,native_sr = librosa.core.load(self.wavfile,duration=2,sr=None, mono=True)

# Resample and save if necessary 
if int(native_sr)!=int(self.sample_rate):
    print("Native Sampling rate is {}".format(native_sr))

    print('Resampling to {sr} Hz'.format(sr=self.sample_rate))

    wav,_ = librosa.core.load(self.wavfile, sr=self.sample_rate, mono=True)
    soundfile.write(self.wavfile,wav,self.sample_rate)