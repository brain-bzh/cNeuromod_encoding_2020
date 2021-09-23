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

# all_early_stopping_criterions = {
#     'train_loss':0, 
#     'train_r2_max':1,
#     'train_r2_mean':2,
#     'val_loss':3, 
#     'val_r2_max':4,
#     'val_r2_mean':5
# }

soundNet_params_path = './sound8.pth' #'/home/maellef/git_dir/cNeuromod_encoding_2020/sound8.pth'
mistroifile = '/home/maelle/Database/fMRI_parcellations/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'

def model_training(outpath, data_selection, data_processing, training_hyperparameters, ml_analysis):
    # WIP CHECK ---> still needed ?
    checkpt_still_here = os.path.lexists('checkpoint.pt') #'/home/maellef/scratch/checkpoint.pt'
    if checkpt_still_here : 
        print('suppression of checkpoint file')
        os.remove('checkpoint.pt') #'/home/maellef/scratch/checkpoint.pt'

    #data selection
    all_subs_files = data_selection['all_data']
    subject = data_selection['subject']
    train_data = []
    for subdata in data_selection['train_data']:
        train_data.extend(all_subs_files[subdata]) 
    eval_data = []
    for subdata in data_selection['eval_data']:
        eval_data.extend(all_subs_files[subdata])
    sessions_train = data_selection['sessions_train']
    sessions_eval = data_selection['sessions_eval']

    #data processing
    scale = data_processing['scale']
    tr = data_processing['tr']
    sr = data_processing['sr']
    selected_inputs =data_processing['selected_inputs']
    if selected_inputs != None : 
        nInputs = len(selected_inputs)
    elif scale == 'MIST_ROI':
        nInputs = 210
    elif scale == 'auditory_Voxels':
        nInputs = 556

    #training_parameters
    model = training_hyperparameters['model']
    fmrihidden = training_hyperparameters['fmrihidden']
    kernel_size = training_hyperparameters['kernel_size']
    finetune_start = training_hyperparameters['finetune_start'] 
    output_layer = training_hyperparameters['output_layer']
    patience_es = training_hyperparameters['patience']
    delta_es = training_hyperparameters['delta']
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
    
    #WIP_conditions :
    #train_pass = training_hyperparameters['train_pass']
    #warm_restart = training_hyperparameters['warm_restart']

    #----------define-paths-and-names----------------------
    outfile_name = str(scale)+'_'+str(model.__name__)+'_'+str(fmrihidden)+'_ks'+str(kernel_size)+'_lr'+str(lr)+'_wd'+str(weight_decay)+'_'
    outfile_name = outfile_name+'decoupled_' if decoupled_weightDecay else outfile_name
    outfile_name = outfile_name+'lrSch_' if lr_scheduler else outfile_name
    outfile_name = outfile_name+'pt_' if power_transform else outfile_name
    #WIP : outfile_name = outfile_name+'trainPass_' if train_pass else outfile_name
    destdir = outpath

    #------------------select data (dataset, films, sessions/seasons)

    #input : list of tuple of shape (audio_path, scan_path)
    train_input = [(data[0], data[sessions_train]) for data in train_data]
    eval_input = [(data[0], data[sessions_eval]) for data in eval_data]

    DataTrain, DataVal, DataTest = create_train_eval_dataset(train_input, eval_input, train_percent, val_percent, test_percent)

    xTrain, yTrain = create_usable_audiofmri_datasets(DataTrain, tr, sr, name='training')
    TrainDataset = SequentialDataset(xTrain, yTrain, batch_size=batchsize, selection=selected_inputs)
    trainloader = DataLoader(TrainDataset, batch_size=None)

    xVal, yVal = create_usable_audiofmri_datasets(DataVal, tr, sr, name='validation')
    ValDataset = SequentialDataset(xVal, yVal, batch_size=batchsize, selection=selected_inputs)
    valloader = DataLoader(ValDataset, batch_size=None)

    xTest, yTest = create_usable_audiofmri_datasets(DataTest, tr, sr, name='test')
    TestDataset = SequentialDataset(xTest, yTest, batch_size=batchsize, selection=selected_inputs)
    testloader = DataLoader(TestDataset, batch_size=None)

    print(f'size of train, val and set : ', len(TrainDataset), len(ValDataset), len(TestDataset))

    #|--------------------------------------------------------------------------------------------------------------------------------------
    ### Model Setup
    print(f'nInputs : ', nInputs, ', kernel size : ', kernel_size, ', output_layer : ', output_layer, ', finetune_start : ', finetune_start,
            ' power_transform : ', power_transform, ', weight_decay', weight_decay)
    net = encod.SoundNetEncoding_conv(pytorch_param_path=soundNet_params_path,fmrihidden=fmrihidden,out_size=nInputs, output_layer=output_layer,
                                    kernel_size=kernel_size, power_transform=power_transform, train_start= finetune_start)
    if gpu : 
        net.to("cuda")
    else:
        net.to("cpu")

    if decoupled_weightDecay : 
        optimizer = optim.AdamW(net.parameters(), lr = lr, weight_decay=weight_decay)
    elif not decoupled_weightDecay : 
        optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay=weight_decay)

    if lr_scheduler : 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    early_stopping = EarlyStopping(patience=patience_es, verbose=True,delta=delta_es)

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

            t_l, t_r2 = train(trainloader,net,optimizer,mseloss=mseloss, gpu=gpu)
            train_loss.append(t_l)
            train_r2_max.append(max(t_r2))
            train_r2_mean.append(np.mean(t_r2))

            v_l, v_r2 = test(valloader,net,optimizer,mseloss=mseloss, gpu=gpu)
            val_loss.append(v_l)
            val_r2_max.append(max(v_r2))
            val_r2_mean.append(np.mean(v_r2))
            
            lrs.append(optimizer.param_groups[0]["lr"])

            print("Train Loss {} Train Mean R2 :  {} Train Max R2 : {}, Val Loss {} Val Mean R2:  {} Val Max R2 : {} ".format(train_loss[-1],train_r2_mean[-1],train_r2_max[-1],val_loss[-1],val_r2_mean[-1],val_r2_max[-1]))

            if ml_analysis == 'wandb':
                wandb.log({"train loss": t_l, "train r2 max": max(t_r2), "train r2 mean":np.mean(t_r2),
                            "val loss": v_l, "val r2 max": max(v_r2), "val r2 mean":np.mean(v_r2),
                            "learning rate" : optimizer.param_groups[0]["lr"], "nb epochs": epoch
                })
            elif ml_analysis == 'comet':
                pass
            else : 
                pass

            if lr_scheduler : 
                scheduler.step(v_l)

            early_stopping(v_l, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
        
    # WIP : 
    # if train_pass:
    #     test(1,testloader,net,optimizer,mseloss=mseloss, gpu=gpu)
    enddate = datetime.now()
    test_loss, final_model = test(testloader,net,optimizer,mseloss=mseloss, gpu=gpu)
    print("Test Loss : {}".format(test_loss))
    
    if ml_analysis == 'wandb':
        wandb.log({"test loss": test_loss, "test r2 max": max(final_model), "test r2 mean":np.mean(final_model)})
    elif ml_analysis == 'comet':
        pass
    else : 
        pass

    #6 - Save Model

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
                'selected ROI': selected_inputs,
                'lrs' : lrs
            }

    ### Nifti file Save
    # if scale == 'MIST_ROI' and nInputs == 210:
    #     r2_img = signals_to_img_labels(r2model.reshape(1,-1),mistroifile)
    #     r2_img.to_filename(str_bestmodel_nii)
    save(state, str_bestmodel)


#---------WIP------------------------------------
    checkpt_still_here = os.path.lexists('checkpoint.pt')
    if checkpt_still_here : 
        print('suppression of checkpoint file')
        os.remove('checkpoint.pt')

#------------------------------------------------------------------------------------------------------------------------------------
#training_called_by_bash

if __name__ == "__main__":
    date = datetime.now()
    dt_string = date.strftime("%Y%m%d")

    #bash command example : python  model_training.py -s 01 -d friends --trainData s01 --evalData s02 --scale auditory_Voxels
    #bash command example : python  model_training.py -s 01 -d movie10 --trainData wolf --evalData bourne --scale MIST_ROI

    parser = argparse.ArgumentParser()

    #data_selection
    parser.add_argument("-s", "--sub", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--sessions_train", type=int, default=1) # WIP, must be >=1, add a condition to check the entry
    parser.add_argument("--sessions_eval", type=int, default=1) # WIP, must be >=1, add a condition to check the entry
    parser.add_argument("--trainData", type=str, nargs='+')
    parser.add_argument("--evalData", type=str, nargs='+')

    #data_processing
    parser.add_argument("--scale", type=str)
    parser.add_argument("--select", type=int, nargs='+', default=None) # in case we want to learn on specific ROI/Voxels
    parser.add_argument("--tr", type=float, default=1.49)
    parser.add_argument("--sr", type=int, default=22050)

    #model_parameters
    parser.add_argument("--hs", type=int, default=1000) #not used anymore with the conv as encoding layer, should we keep it ?
    parser.add_argument("--bs", type=int, default=30)
    parser.add_argument("--ks", type=int, default=5)
    parser.add_argument("-f","--finetuneStart", type=str, default=None) #among "conv1", "pool1", ... "conv8", "conv8_2"
    parser.add_argument("-o", "--outputLayer", type=str, default="conv7")#output layer

    #training_hyperparameters
        #early_stopping
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--delta", type=float, default=0)
        #dataset_size    
    parser.add_argument("--train100", type=float, default=0.6)
    parser.add_argument("--test100", type=float, default=0.2)
    parser.add_argument("--val100", type=float, default=0.2)
        #other
    parser.add_argument("--gpu", dest='gpu', action='store_true')
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--nbepoch", type=int, default=200)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--decoupledWD", dest='decoupledWD', action='store_true')
    parser.add_argument("--powerTransform", dest='powerTransform', action='store_true')
    parser.add_argument("--lrScheduler", dest='lrScheduler', action='store_true')
    # WIP : parser.add_argument("--trainPass", dest='trainPass', action='store_true')

    #ML_analysis
    parser.add_argument("--wandb", dest='wandb', action='store_true')
    parser.add_argument("--comet", dest='comet', action='store_true')

    args = parser.parse_args()

    data_selection = {
        'subject' : int(args.sub),
        'dataset' : args.dataset,
        'train_data' : args.trainData,
        'eval_data' : args.evalData,
        'sessions_train' : args.sessions_train,
        'sessions_eval' : args.sessions_eval
    }
    ds = data_selection

    data_processing = {
        'scale': args.scale,
        'tr' : args.tr,
        'sr' : args.sr,
        'selected_inputs': args.select                     
    }
    dp = data_processing

    training_hyperparameters = {
        'model':encod.SoundNetEncoding_conv,
        'mseloss':nn.MSELoss(reduction='sum'),
        'fmrihidden':args.hs,
        'batchsize':args.bs,
        'finetune_start':args.finetuneStart,
        'output_layer':args.outputLayer,
        'kernel_size':args.ks,
        'patience':args.patience,
        'delta':args.delta, 
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
        # WIP : 'train_pass' : args.trainPass
    }
    th = training_hyperparameters

    #-------------------------------------------------------------
    ml_analysis = ''
    if args.wandb :
        #os.environ['WANDB_MODE'] = 'offline' #for beluga environment
        print("wandb")
        import wandb 
        wandb.init(project="neuroencoding_audio", config={})
        wandb.config.update(args)
        print("update config okay")
        ml_analysis += 'wandb'

    elif args.comet : 
        import comet_ml
        experiment = comet_ml.Experiment("1NT8FqmXsAH088rHLBYC1Yyev")
        ml_analysis += 'comet'

    outpath = '/home/maelle/Results/' #"/home/maellef/Results/"
    stimuli_path = '/home/maelle/DataBase/stimuli' #'/home/maellef/DataBase/stimuli'
    embed_path = '/home/maelle/DataBase/fMRI_Embeddings' #'/home/maellef/DataBase/fMRI_Embeddings'
    
    dataset_path = os.path.join(stimuli_path, ds['dataset'])
    parcellation_path = os.path.join(embed_path, dp['scale'], ds['dataset'], 'sub-'+args.sub)

    all_subs_files = dict()
    for film in os.listdir(dataset_path):
        film_path = os.path.join(dataset_path, film)
        if os.path.isdir(film_path):
            all_subs_files[film] = fu.associate_stimuli_with_Parcellation(film_path, parcellation_path)

    resultpath = outpath+dt_string+"verify_train_{}_lr0.01_{}epochs".format(dp['scale'], th['nbepoch'])
    resultpath = os.path.join(resultpath, 'sub-'+args.sub)
    os.makedirs(resultpath, exist_ok=True)
    
    ds['all_data']=all_subs_files
    model_training(resultpath, ds, dp, th, ml_analysis)

    if args.wandb :
        wandb.finish()
    elif args.comet : 
        pass

