import torch
import numpy as np
from sklearn.metrics import r2_score

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_path='/home/maellef/scratch/checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_path=checkpoint_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter > self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss


def train(trainloader,net,optimizer, epoch, mseloss,delta=1e-2, gpu=True):
    all_y = []
    all_y_predicted = []
    running_loss = 0
    net.train()

    for batch_nb, (x, y) in enumerate(trainloader):
        #print(f'batch n°', batch_nb)
        optimizer.zero_grad()
        batch_size = x.shape[0]
        #print(f'   batch size (x.shape[0]) : ', batch_size)

        # for 1D output
        #print(f'x before becoming into a tensor and reshaping : ', type(x2), x2.shape)
        x =  torch.Tensor(x).view(1,1,-1, 1)          #we need to adapt the shape of the batch so it can fit into pytorch convolutions layers
        #print(f'x before becoming into a tensor and reshaping : ', type(x2), x2.shape)
        if gpu : 
            x = x.cuda()                                           #then we put the tensor into the memory of the graphic card, so computations can be done faster

        # Forward pass
        predicted_y = net(x, epoch)
        predicted_y = predicted_y.permute(2,1,0,3).squeeze().double()# as some dimensions in the output 
        #print(f'   len(predicted_y) : ', len(predicted_y))        
        
        predicted_y = predicted_y[:batch_size]                #FOR AUDIO ONLY : Cropping the end of the predicted fmri to match the measured bold
        #print(f'predicted_y after squeezing and cutting to the same size as fmri data: ', predicting_y)
        #print(f'   len(predicted_y) after crop: ', len(predicted_y))
        
        y = y.double()
        #print(f'   len(real_y): ', len(y))
        if gpu:
            y = y.cuda()
        #print(f"y_real shape : ", y.shape, "and y_predicted shape : ", predicted_y.shape)         # both must have the same shape
        loss=delta*mseloss(predicted_y,y)/batch_size
        loss.backward()
        optimizer.step()

        all_y.append(y.cpu().numpy().reshape(batch_size,-1))
        all_y_predicted.append(predicted_y.detach().cpu().numpy().reshape(batch_size,-1))
        running_loss += loss.item()
        
    r2_model = r2_score(np.vstack(all_y),np.vstack(all_y_predicted),multioutput='raw_values') 
    return running_loss/batch_nb, r2_model

def train_without_grad(trainloader,net,optimizer,mseloss,delta=1e-2, gpu=True):
    all_y = []
    all_y_predicted = []
    running_loss = 0
    net.train()

    with torch.no_grad() : 
        for batch_nb, (x, y) in enumerate(trainloader):
            print(f'batch n°', batch_nb)
            optimizer.zero_grad()
            batch_size = x.shape[0]
            print(f'   batch size (x.shape[0]) : ', batch_size)

            # for 1D output 
            x =  torch.Tensor(x).view(1,1,-1, 1) 
            if gpu:
                x = x.cuda()  
            # Forward pass
            predicted_y = net(x)
            predicted_y = predicted_y.permute(2,1,0,3).squeeze().double()
            print(f'   len(predicted_y) : ', len(predicted_y))
            predicted_y = predicted_y[:batch_size]
            print(f'   len(predicted_y) after crop: ', len(predicted_y))
            y = y.double()
            print(f'   len(real_y): ', len(y))
            if gpu:
                y = y.cuda()

            loss=delta*mseloss(predicted_y,y)/batch_size

            all_y.append(y.cpu().numpy().reshape(batch_size,-1))
            all_y_predicted.append(predicted_y.detach().cpu().numpy().reshape(batch_size,-1))
            running_loss += loss.item()
            

        r2_model = r2_score(np.vstack(all_y),np.vstack(all_y_predicted),multioutput='raw_values') 
        return running_loss/batch_nb, r2_model


def test(trainloader,net,optimizer, epoch ,mseloss,delta=1e-2, gpu=True):
    all_y = []
    all_y_predicted = []
    running_loss = 0
    net.eval()

    with torch.no_grad() : 
        for batch_nb, (x, y) in enumerate(trainloader):
            optimizer.zero_grad()
            batch_size = x.shape[0]

            # for 1D output 
            x =  torch.Tensor(x).view(1,1,-1, 1) 
            if gpu:
                x = x.cuda()  
            # Forward pass
            predicted_y = net(x, epoch)
            predicted_y = predicted_y.permute(2,1,0,3).squeeze().double()
            predicted_y = predicted_y[:batch_size]
            y = y.double()
            if gpu:
                y = y.cuda()

            loss=delta*mseloss(predicted_y,y)/batch_size

            all_y.append(y.cpu().numpy().reshape(batch_size,-1))
            all_y_predicted.append(predicted_y.detach().cpu().numpy().reshape(batch_size,-1))
            running_loss += loss.item()
            

        r2_model = r2_score(np.vstack(all_y),np.vstack(all_y_predicted),multioutput='raw_values') 
        return running_loss/batch_nb, r2_model


def test_r2(testloader,net, epoch, mseloss, gpu=True):
    all_fmri = []
    all_fmri_p = []
    net.eval()
    with torch.no_grad():
        for (wav,fmri) in testloader:

            bsize = wav.shape[0]
            
            # load data
            wav = torch.Tensor(wav).view(1,1,-1,1)
            if gpu:
                wav = wav.cuda()

            fmri = fmri.view(bsize,-1)
            if gpu:
                fmri=fmri.cuda()

            # Forward pass
            fmri_p = net(wav, epoch).permute(2,1,0,3).squeeze()

            #Cropping the end of the predicted fmri to match the measured bold
            fmri_p = fmri_p[:bsize]
            
            all_fmri.append(fmri.cpu().numpy().reshape(bsize,-1))
            all_fmri_p.append(fmri_p.cpu().numpy().reshape(bsize,-1))

    r2_model = r2_score(np.vstack(all_fmri),np.vstack(all_fmri_p),multioutput='raw_values')
    return r2_model