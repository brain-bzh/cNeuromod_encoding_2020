#Generic
import os, warnings, argparse
import numpy as np
from datetime import datetime
#Get input & preprocess
import files_utils as fu
#Create Dataset
from random import sample
from torch.utils.data import DataLoader
from Datasets_utils import SequentialDataset, create_usable_audiofmri_datasets
#Models
from models import encoding_models as encod
#Train & Test
from tqdm import tqdm
from torch import nn, optim, save
from train_utils import train, test, test_r2, EarlyStopping
#Visualisation
from nilearn.plotting import plot_stat_map
from nilearn.regions import signals_to_img_labels
from matplotlib import pyplot as plt 

all_films = {
    'bourne':'bourne_supremacy',
    'wolf':'wolf_of_wall_street',
    'life':'life',
    'hidden':'hidden_figures',
}

all_criterions = {
    'train_loss':0, 
    'train_r2_max':1,
    'train_r2_mean':2,
    'val_loss':3, 
    'val_r2_max':4,
    'val_r2_mean':5
}

def model_training(outpath, data_selection, data_processing, training_hyperparameters):
    checkpt_still_here = os.path.lexists('checkpoint.pt')
    if checkpt_still_here : 
        print('suppression of checkpoint file')
        os.remove('checkpoint.pt')

    #define arguments
    all_subs_files = data_selection['all_data']
    subject = data_selection['subject']
    film = data_selection['film']
    sessions = data_selection['sessions']

    scale = data_processing['scale']
    tr = data_processing['tr']
    sr = data_processing['sr']
    selected_ROI =data_processing['selected_ROI']
    nInputs = data_processing['nInputs']

    diff_TrainTest = training_hyperparameters['diff_sess_for_train_test']
    model = training_hyperparameters['model']
    fmrihidden = training_hyperparameters['fmrihidden']
    kernel_size = training_hyperparameters['kernel_size']
    gpu = training_hyperparameters['gpu']
    batchsize = training_hyperparameters['batchsize']
    lr = training_hyperparameters['lr']
    weight_decay = training_hyperparameters['weight_decay']
    nbepoch = training_hyperparameters['nbepoch']
    train_percent = training_hyperparameters['train_percent']
    test_percent = training_hyperparameters['test_percent']
    val_percent = training_hyperparameters['val_percent']
    mseloss = training_hyperparameters['mseloss']
    decoupled_weightDecay = training_hyperparameters['decoupled_weightDecay']
    power_transform = training_hyperparameters['power_transform']
    #es_criterion = training_hyperparameters['early_stopping']

    #warm_restart = training_hyperparameters['warm_restart'] àtester
    #early_stopping = training_hyperparameters['early_stopping']
    #problem, that stop the training of any following test
    #to look how to implement a more interactable early_stopping

    #----------define-paths-and-names----------------------
    outfile_name = str(film)+'_'+str(scale)+str(nInputs)+'_'+str(model.__name__)+'_'+str(fmrihidden)+'_ks'+str(kernel_size)+'_lr'+str(lr)+'_wd'+str(weight_decay)+'_'
    outfile_name = outfile_name+'decoupled_' if decoupled_weightDecay else outfile_name
    outfile_name = outfile_name+'pt_' if power_transform else outfile_name
    destdir = outpath

    #------------------select data (dataset, films, sessions/seasons)
    
    all_data = all_subs_files
    
    print(all_data)

    #----------WIP----------------------------------
    DataTest = []
    DataTrain = []
    for wav_fmri_pair in all_data:
        wav = wav_fmri_pair[0]
        fmri_sesvid = wav_fmri_pair[1:]
        if len(fmri_sesvid)>1 and diff_TrainTest:
            DataTrain.append((wav, fmri_sesvid[0]))
            DataTest.append((wav, fmri_sesvid[1]))
        elif wav.find('s2e') > -1:
            DataTest.append((wav, fmri_sesvid[0]))
        elif wav.find('s1e') : 
            DataTrain.append((wav, fmri_sesvid[0]))

    print(f'lenght of DataTrain : ', len(DataTrain))
    print(f'lenght of DataTest : ', len(DataTest))

    xTrain, yTrain = create_usable_audiofmri_datasets(DataTrain, tr, sr, name='training')
    TrainDataset = SequentialDataset(xTrain, yTrain, batch_size=batchsize, selection=selected_ROI)
    total_train_len = len(TrainDataset)
    train_len = int(np.floor(train_percent*total_train_len))
    #if same dataset for training and testing
    if len(DataTest)==0:
        val_len = int(np.floor(val_percent*total_train_len))
        test_len = total_train_len-train_len-val_len
    #if separete datasets for training and testing
    else:
        xTest, yTest = create_usable_audiofmri_datasets(DataTest, tr, sr, name='test')
        TestDataset = SequentialDataset(xTest, yTest, batch_size=batchsize, selection=selected_ROI)
        total_test_len = len(TestDataset)
        val_len = int(np.floor(val_percent*total_test_len))
        test_len = total_test_len-val_len

    print(f'size of train, val and set : ', train_len, test_len, val_len)

    loader = list(DataLoader(TrainDataset, batch_size=None))
    trainloader = sample(loader[:train_len], k=train_len)

    if len(DataTest)==0:
        valloader = sample(loader[train_len:train_len+val_len], k=val_len)
        testloader = sample(loader[train_len+val_len:train_len+val_len+test_len], k=test_len)
    else : 
        second_loader = list(DataLoader(TestDataset, batch_size=None))
        valloader = sample(second_loader[:val_len], k=val_len)
        testloader = sample(second_loader[val_len:], k=test_len)

    #|--------------------------------------------------------------------------------------------------------------------------------------
    ### Model Setup
    net = encod.SoundNetEncoding_conv(pytorch_param_path='./sound8.pth',fmrihidden=fmrihidden,out_size=nInputs, kernel_size=kernel_size, power_transform=power_transform)
    if gpu : 
        net.to("cuda")
    else:
        net.to("cpu")

    if decoupled_weightDecay : 
        optimizer = optim.AdamW(net.parameters(), lr = lr, weight_decay=weight_decay)
    elif not decoupled_weightDecay : 
        optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=True,delta=0)
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
    criterions = [train_loss, -train_r2_max[-1], -train_r2_mean[-1], val_loss, -val_r2_max[-1], -val_r2_mean[-1]]


    try:
        for epoch in tqdm(range(nbepoch)):
            t_l, t_r2 = train(epoch,trainloader,net,optimizer,mseloss=mseloss, gpu=gpu)
            train_loss.append(t_l)
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
            early_stopping(criterions[es_criterion], net)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    test_loss, final_model = test(1,testloader,net,optimizer,mseloss=mseloss, gpu=gpu)
    print("Test Loss : {}".format(test_loss))

    #6 - Save Model

    mistroifile = '/home/maelle/Database/fMRI_parcellations/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'

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
    if scale == 'MIST_ROI' and nInputs == 210:
        r2_img = signals_to_img_labels(r2model.reshape(1,-1),mistroifile)
        r2_img.to_filename(str_bestmodel_nii)
    save(state, str_bestmodel)

    checkpt_still_here = os.path.lexists('checkpoint.pt')
    if checkpt_still_here : 
        print('suppression of checkpoint file')
        os.remove('checkpoint.pt')

#------------------------------------------------------------------------------------------------------------------------------------
#training_called_by_bash

if __name__ == "__main__":
    date = datetime.now()
    dt_string = date.strftime("%Y%m%d")

    #bash command example : python  model_training.py -s 01 -d <friends, movie10> -f <None, Bourne, Hidden, Life, Wolf, All> --scale MIST_ROI --nInputs 210 --train100 1.0 --test100 0.5 --val100 0.5
    parser = argparse.ArgumentParser()
    #data_selection
    parser.add_argument("-s", "--sub", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-f", "--film", type=str, default='')
    parser.add_argument("--sessions", type=str, nargs='+', default=None)
    #data_processing
    parser.add_argument("--scale", type=str)
    parser.add_argument("--nInputs", type=int)
    parser.add_argument("--tr", type=float, default=1.49)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--select", type=int, nargs='+', default=None)
    #model_parameters
    parser.add_argument("--hidden_size", type=int, default=1000)
    parser.add_argument("--bs", type=int, default=30)
    parser.add_argument("--ks", type=int, default=5)
    #training_hyperparameters
    parser.add_argument("--es", type=str)
    parser.add_argument("--train100", type=float)
    parser.add_argument("--test100", type=float)
    parser.add_argument("--val100", type=float)
    parser.add_argument("--gpu", dest='gpu', action='store_true')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--nbepoch", type=int, default=200)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--decoupledWD", dest='decoupledWD', action='store_true')
    parser.add_argument("--powerTransform", dest='powerTransform', action='store_true')

    args = parser.parse_args()
    data_selection = {
        'subject':int(args.sub),
        'dataset':args.dataset,
        'film':args.film,
        'sessions':args.sessions
    }
    ds = data_selection

    data_processing = {
        'scale': args.scale,
        'tr' : args.tr,
        'sr' : args.sr,
        'selected_ROI': args.select,
        'nInputs': args.nInputs                          
    }
    dp = data_processing

    training_hyperparameters = {
        'model':encod.SoundNetEncoding_conv,
        'diff_sess_for_train_test':True,
        'mseloss':nn.MSELoss(reduction='sum'),

        #'early_stopping':all_criterions[args.es],
        'fmrihidden':args.hidden_size,
        'batchsize':args.bs,
        'kernel_size':args.ks,
        'train_percent':args.train100,
        'test_percent':args.test100,
        'val_percent':args.val100,
        'gpu':args.gpu,
        'lr':args.lr,
        'nbepoch': args.nbepoch,
        'weight_decay':args.wd,
        'decoupled_weightDecay' : args.decoupledWD,
        'power_transform' : args.powerTransform
    }
    th = training_hyperparameters

    #-------------------------------------------------------------

    outpath = '/home/maelle/Results/' #"/home/maellef/Results/"
    stimuli_path = '/home/maelle/DataBase/stimuli' #'/home/maellef/DataBase/stimuli'
    embed_path = '/home/maelle/DataBase/fMRI_Embeddings' #'/home/maellef/DataBase/fMRI_Embeddings'
    
    dataset_path = os.path.join(stimuli_path, ds['dataset'], ds['film'])
    parcellation_path = os.path.join(embed_path, dp['scale'], ds['dataset'], 'sub-'+args.sub)

    if ds['film'] == '' :
        all_subs_files = list()
        for film in os.listdir(dataset_path):
            film_path = os.path.join(dataset_path, film)
            all_subs_files.append(fu.associate_stimuli_with_Parcellation(film_path, parcellation_path))
        ds['film'] = 'all'
    else : 
        all_subs_files = fu.associate_stimuli_with_Parcellation(dataset_path, parcellation_path)
    
    print(all_subs_files)

    resultpath = outpath+dt_string+"test_{}{}_lr0.01_{}epochs".format(dp['scale'],dp['nInputs'], th['nbepoch'])
    resultpath = os.path.join(resultpath, 'sub-'+args.sub)
    os.makedirs(resultpath, exist_ok=True)
    
    ds['all_data']=all_subs_files
    model_training(resultpath, ds, dp, th)

