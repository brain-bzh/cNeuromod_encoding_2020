#Generic
import os
import warnings
import numpy as np
from datetime import datetime
#Get input & preprocess
from files_utils import associate_stimuli_with_Parcellation, create_dir_if_needed
from audio_utils import convert_Audio
#Create Dataset
from itertools import islice
from random import shuffle, sample
from torch.utils.data import DataLoader
from Datasets_utils import SequentialDataset
from audio_utils import load_audio_by_bit
import librosa
#Models
from models import soundnet_model as sdn
from models import encoding_models as encod
#Train & Test
from tqdm import tqdm
from torch import nn, optim, save
from train_utils import train, test, test_r2, EarlyStopping
#Visualisation
from nilearn.plotting import plot_stat_map
from nilearn.regions import signals_to_img_labels
from matplotlib import pyplot as plt 

def main_model_training(outpath, data_selection, data_processing, training_hyperparameters):
    checkpt_still_here = os.path.lexists('checkpoint.pt')
    if checkpt_still_here : 
        print('suppression of checkpoint file')
        os.remove('checkpoint.pt')

    #define arguments
    all_subs_files = data_selection['all_data']
    subject = data_selection['subject']
    film = data_selection['film']

    scale = data_processing['scale']
    tr = data_processing['tr']
    sr = data_processing['sr']
    selected_ROI =data_processing['selected_ROI']
    nroi = data_processing['nroi']

    model = training_hyperparameters['model']
    fmrihidden = training_hyperparameters['fmrihidden']
    kernel_size = training_hyperparameters['kernel_size']
    gpu = training_hyperparameters['gpu']
    batchsize = training_hyperparameters['batchsize']
    lr = training_hyperparameters['lr']
    nbepoch = training_hyperparameters['nbepoch']
    train_percent = training_hyperparameters['train_percent']
    test_percent = training_hyperparameters['test_percent']
    val_percent = training_hyperparameters['val_percent']
    mseloss = training_hyperparameters['mseloss']
    #early_stopping = training_hyperparameters['early_stopping']
    #problem, that stop the training of any following test
    #to look how to implement a more interactable early_stopping

    outfile_name = str(scale)+'_'+str(model.__name__)+'_'+str(fmrihidden)+'_ks_'+str(kernel_size)+'_lr_'+str(lr)+'_'
    destdir = outpath

    #-------------------------------------------------------------
    DataTest = all_subs_files[subject][film]

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

    dataset = SequentialDataset(x, y, batch_size=batchsize, selection=selected_ROI)

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

    #|--------------------------------------------------------------------------------------------------------------------------------------
    ### Model Setup
    net = encod.SoundNetEncoding_conv(pytorch_param_path='./sound8.pth',fmrihidden=fmrihidden,out_size=nroi, kernel_size=kernel_size)
    if gpu : 
        net.to("cuda")
    else:
        net.to("cpu")

    optimizer = optim.Adam(net.parameters(), lr = lr)
    early_stopping = EarlyStopping(patience=10, verbose=True,delta=1e-3)
    enddate = datetime.now()

    #---------------------------------------------------------------------------------------------------------------------------------
    #5 - Train & Test

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
            t_l, t_r2 = train(epoch,trainloader,net,optimizer,mseloss=mseloss, gpu=gpu)
            train_loss.append(t_l)
            print(t_r2)
            train_r2_max.append(max(t_r2))
            train_r2_mean.append(np.mean(t_r2))

            v_l, v_r2 = test(epoch,valloader,net,optimizer,mseloss=mseloss, gpu=gpu)
            val_loss.append(v_l)
            val_r2_max.append(max(v_r2))
            val_r2_mean.append(np.mean(v_r2))
            print("Train Loss {} Train Mean R2 :  {} Train Max R2 : {}, Val Loss {} Val Mean R2:  {} Val Max R2 : {} ".format(train_loss[-1],train_r2_mean[-1],train_r2_max[-1],val_loss[-1],val_r2_mean[-1],val_r2_max[-1]))

            # early_stopping needs the R2 mean to check if it has increased, 
            # and if it has, it will make a checkpoint of the current model
            
            #r2_forEL = -(val_r2_max[-1])
            early_stopping(t_l, net)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    test_loss = test(1,testloader,net,optimizer,mseloss=mseloss, gpu=gpu)
    print("Test Loss : {}".format(test_loss))

    #6 - Save Model

    mistroifile = '/home/maelle/Database/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'

    dt_string = enddate.strftime("%Y-%m-%d-%H-%M-%S")
    outfile_name += dt_string

    str_bestmodel = os.path.join(destdir,"{}.pt".format(outfile_name))
    str_bestmodel_nii = os.path.join(destdir,"{}.nii.gz".format(outfile_name))

    r2model = test_r2(testloader,net,mseloss, gpu=gpu)
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
                'test_r2' : r2model,
                'test_r2_max' : r2model.max(),
                'test_r2_mean' : r2model.mean(),
                'training_time' : enddate - startdate,
                'nhidden' : fmrihidden,
                'model' : net,
                'selected ROI': selected_ROI
            }

    ### Nifti file Save
    if scale == 'MIST_ROI' and nroi == 210:
        r2_img = signals_to_img_labels(r2model.reshape(1,-1),mistroifile)
        r2_img.to_filename(str_bestmodel_nii)
    save(state, str_bestmodel)

    checkpt_still_here = os.path.lexists('checkpoint.pt')
    if checkpt_still_here : 
        print('suppression of checkpoint file')
        os.remove('checkpoint.pt')
