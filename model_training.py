#Generic
import os, warnings, argparse
import numpy as np
from datetime import datetime
#Get input & preprocess
import files_utils as fu
#Create Dataset
from random import sample
from torch.utils.data import DataLoader
from Datasets_utils import SequentialDataset, create_usable_audiofmri_datasets, create_train_eval_dataset
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
    train_data = all_subs_files[data_selection['train_data']]
    eval_data = all_subs_files[data_selection['eval_data']]
    #film = data_selection['film']/
    sessions_train = data_selection['sessions_train']
    sessions_eval = data_selection['sessions_eval']

    scale = data_processing['scale']
    tr = data_processing['tr']
    sr = data_processing['sr']
    selected_ROI =data_processing['selected_ROI']
    if scale == 'MIST_ROI':
        nInputs = 210
    elif scale == 'auditory_Voxels':
        nInputs = 556

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
    lr_scheduler = training_hyperparameters['lr_scheduler']
    train_pass = training_hyperparameters['train_pass']

    #es_criterion = training_hyperparameters['early_stopping']

    #warm_restart = training_hyperparameters['warm_restart'] àtester
    #early_stopping = training_hyperparameters['early_stopping']
    #problem, that stop the training of any following test
    #to look how to implement a more interactable early_stopping

    #----------define-paths-and-names----------------------
    outfile_name = str(scale)+'_'+str(model.__name__)+'_'+str(fmrihidden)+'_ks'+str(kernel_size)+'_lr'+str(lr)+'_wd'+str(weight_decay)+'_'
    outfile_name = outfile_name+'decoupled_' if decoupled_weightDecay else outfile_name
    outfile_name = outfile_name+'trainPass_' if train_pass else outfile_name
    outfile_name = outfile_name+'lrSch_' if lr_scheduler else outfile_name
    outfile_name = outfile_name+'pt_' if power_transform else outfile_name
    destdir = outpath

    #------------------select data (dataset, films, sessions/seasons)

    train_input = [(data[0], data[sessions_train]) for data in train_data]
    eval_input = [(data[0], data[sessions_eval]) for data in eval_data]

    DataTrain, DataVal, DataTest = create_train_eval_dataset(train_input, eval_input, train_percent, val_percent, test_percent)

    xTrain, yTrain = create_usable_audiofmri_datasets(DataTrain, tr, sr, name='training')
    TrainDataset = SequentialDataset(xTrain, yTrain, batch_size=batchsize, selection=selected_ROI)
    trainloader = DataLoader(TrainDataset, batch_size=None)

    xVal, yVal = create_usable_audiofmri_datasets(DataVal, tr, sr, name='validation')
    ValDataset = SequentialDataset(xVal, yVal, batch_size=batchsize, selection=selected_ROI)
    valloader = DataLoader(ValDataset, batch_size=None)

    xTest, yTest = create_usable_audiofmri_datasets(DataTest, tr, sr, name='test')
    TestDataset = SequentialDataset(xTest, yTest, batch_size=batchsize, selection=selected_ROI)
    testloader = DataLoader(TestDataset, batch_size=None)

    print(f'size of train, val and set : ', len(TrainDataset), len(ValDataset), len(TestDataset))
    return 0

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

    if lr_scheduler : 
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
    # criterions = [train_loss, -train_r2_max[-1], -train_r2_mean[-1], val_loss, -val_r2_max[-1], -val_r2_mean[-1]]
    lrs = []

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
            lrs.append(optimizer.param_groups[0]["lr"])
            if lr_scheduler : 
                scheduler.step()

            early_stopping(v_l, net)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    
    if train_pass:
        test(1,testloader,net,optimizer,mseloss=mseloss, gpu=gpu)

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
                'selected ROI': selected_ROI,
                'lrs' : lrs
            }

    ### Nifti file Save
    # if scale == 'MIST_ROI' and nInputs == 210:
    #     r2_img = signals_to_img_labels(r2model.reshape(1,-1),mistroifile)
    #     r2_img.to_filename(str_bestmodel_nii)
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

    #bash command example : python  model_training.py -s 01 -d <friends, movie10> -f <None, Bourne, Hidden, Life, Wolf, All> --scale <MIST_ROI, auditory_Voxels>
    #bash command example : python  model_training.py -s 01 -d friends --trainData s01 --evalData s02 --scale auditory_Voxels
    #bash command example : python  model_training.py -s 01 -d movie10 --trainData wolf --evalData bourne --scale MIST_ROI

    parser = argparse.ArgumentParser()

    #data_selection
    parser.add_argument("-s", "--sub", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    #parser.add_argument("-f", "--film", type=str, default='')
    parser.add_argument("--sessions_train", type=int, default=1)
    parser.add_argument("--sessions_eval", type=int, default=1)
    parser.add_argument("--trainData", type=str, default='')
    parser.add_argument("--evalData", type=str, default='')

    #data_processing
    parser.add_argument("--scale", type=str)
    #parser.add_argument("--nInputs", type=int)
    parser.add_argument("--tr", type=float, default=1.49)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--select", type=int, nargs='+', default=None)

    #model_parameters
    parser.add_argument("--hidden_size", type=int, default=1000)
    parser.add_argument("--bs", type=int, default=30)
    parser.add_argument("--ks", type=int, default=5)

    #training_hyperparameters
    parser.add_argument("--es", type=str)
    parser.add_argument("--train100", type=float, default=0.6)
    parser.add_argument("--test100", type=float, default=0.2)
    parser.add_argument("--val100", type=float, default=0.2)
    parser.add_argument("--gpu", dest='gpu', action='store_true')
    parser.add_argument("--lr", type=float, default=10)
    parser.add_argument("--nbepoch", type=int, default=200)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--decoupledWD", dest='decoupledWD', action='store_true')
    parser.add_argument("--powerTransform", dest='powerTransform', action='store_true')
    parser.add_argument("--lrScheduler", dest='lrScheduler', action='store_true')
    parser.add_argument("--trainPass", dest='trainPass', action='store_true')

    args = parser.parse_args()
    data_selection = {
        'subject':int(args.sub),
        'dataset':args.dataset,
        'train_data':args.trainData,
        'eval_data':args.evalData,
        'sessions_train' : args.sessions_train,
        'sessions_eval' : args.sessions_eval
        #'film':args.film,
    }
    ds = data_selection

    data_processing = {
        'scale': args.scale,
        'tr' : args.tr,
        'sr' : args.sr,
        'selected_ROI': args.select
        #'nInputs': args.nInputs                          
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
        'power_transform' : args.powerTransform,
        'lr_scheduler' : args.lrScheduler,
        'train_pass' : args.trainPass
    }
    th = training_hyperparameters

    #-------------------------------------------------------------

    outpath = '/home/maelle/Results/' #"/home/maellef/Results/"
    stimuli_path = '/home/maelle/DataBase/stimuli' #'/home/maellef/DataBase/stimuli'
    embed_path = '/home/maelle/DataBase/fMRI_Embeddings' #'/home/maellef/DataBase/fMRI_Embeddings'
    
    dataset_path = os.path.join(stimuli_path, ds['dataset'])
    parcellation_path = os.path.join(embed_path, dp['scale'], ds['dataset'], 'sub-'+args.sub)

    # if ds['film'] == '' :
    all_subs_files = dict()
    for film in os.listdir(dataset_path):
        film_path = os.path.join(dataset_path, film)
        if os.path.isdir(film_path):
            all_subs_files[film] = fu.associate_stimuli_with_Parcellation(film_path, parcellation_path)

    resultpath = outpath+dt_string+"test_{}_lr0.01_{}epochs".format(dp['scale'], th['nbepoch'])
    resultpath = os.path.join(resultpath, 'sub-'+args.sub)
    os.makedirs(resultpath, exist_ok=True)
    
    ds['all_data']=all_subs_files
    model_training(resultpath, ds, dp, th)

