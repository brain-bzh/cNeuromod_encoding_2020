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


def train(trainloader,net,optimizer, epoch, mseloss,lambada=1e-2, gamma = 1e-4,gpu=True,device="cpu"):
    all_y = []
    all_y_predicted = []
    running_loss = 0
    net.train()
    kl_audio = torch.nn.KLDivLoss(reduction='batchmean')
    for batch_nb, (x, prob, emb) in enumerate(trainloader):
        #print(f'batch n°', batch_nb)
        optimizer.zero_grad()
        batch_size = x.shape[0]
        #print(f'   batch size (x.shape[0]) : ', batch_size)

        # for 1D output
        #print(f'x before becoming into a tensor and reshaping : ', type(x2), x2.shape)        
        x =  torch.Tensor(x).view(1,1,-1, 1)          #we need to adapt the shape of the batch so it can fit into pytorch convolutions layers
        #print(f'x before becoming into a tensor and reshaping : ', type(x2), x2.shape)
        x = x.to(device)
        prob = prob.to(device)
        emb = emb.to(device)
        
        # Forward pass
        predicted_prob,predicted_emb = net(x, epoch)
        
        #print(f"y_real shape : ", y.shape, "and y_predicted shape : ", predicted_y.shape)         # both must have the same shape
        predicted_emb = predicted_emb.permute(2,1,0,3).squeeze()# as some dimensions in the output 
        #print(f'   len(predicted_y) : ', len(predicted_y))        
        
        predicted_emb = predicted_emb[:batch_size]

        predicted_prob = predicted_prob.permute(2,1,0,3).squeeze()
        predicted_prob = predicted_prob[:batch_size]
        #print(f'predicted_y after squeezing and cutting to the same size as fmri data: ', predicting_y)
        #print(f'   len(predicted_y) after crop: ', len(predicted_y))
        #print(f"Predicted prob shape {predicted_prob.shape}, Predicted Embedding shape {predicted_emb.shape} ")
        #print(f"prob shape {prob.shape},  Embedding shape {emb.shape} ")
        #y = y.double()
        #print(f'   len(real_y): ', len(y))

        loss_audioset = kl_audio(torch.nn.functional.log_softmax(predicted_prob,1),prob)
                    
        loss=lambada*mseloss(predicted_emb,emb)/batch_size + gamma*loss_audioset
        loss.backward()
        optimizer.step()

        all_y.append(emb.cpu().numpy().reshape(batch_size,-1))
        all_y_predicted.append(predicted_emb.detach().cpu().numpy().reshape(batch_size,-1))
        running_loss += loss.item()
        
    r2_model = r2_score(np.vstack(all_y),np.vstack(all_y_predicted),multioutput='raw_values') 
    return running_loss/batch_nb, r2_model

def test(trainloader,net,optimizer, epoch ,mseloss,lambada=1e-2, gamma=1e-4,gpu=True,device="cpu"):
    all_y = []
    all_y_predicted = []
    running_loss = 0
    net.eval()
    kl_audio = torch.nn.KLDivLoss(reduction='batchmean')
    with torch.no_grad() : 
        for batch_nb, (x, prob, emb) in enumerate(trainloader):
            optimizer.zero_grad()
            batch_size = x.shape[0]

            x =  torch.Tensor(x).view(1,1,-1, 1)          #we need to adapt the shape of the batch so it can fit into pytorch convolutions layers
            #print(f'x before becoming into a tensor and reshaping : ', type(x2), x2.shape)
            
            x = x.to(device)
            prob = prob.to(device)
            emb= emb.to(device)
            # Forward pass
            predicted_prob,predicted_emb = net(x, epoch)
            
            #print(f"y_real shape : ", y.shape, "and y_predicted shape : ", predicted_y.shape)         # both must have the same shape
            predicted_emb = predicted_emb.permute(2,1,0,3).squeeze()# as some dimensions in the output 
            #print(f'   len(predicted_y) : ', len(predicted_y))        
            
            predicted_emb = predicted_emb[:batch_size]

            predicted_prob = predicted_prob.permute(2,1,0,3).squeeze()
            predicted_prob = predicted_prob[:batch_size]
            #print(f'predicted_y after squeezing and cutting to the same size as fmri data: ', predicting_y)
            #print(f'   len(predicted_y) after crop: ', len(predicted_y))
            #print(f"Predicted prob shape {predicted_prob.shape}, Predicted Embedding shape {predicted_emb.shape} ")
            #print(f"prob shape {prob.shape},  Embedding shape {emb.shape} ")
            #y = y.double()
            #print(f'   len(real_y): ', len(y))

            loss_audioset = kl_audio(torch.nn.functional.log_softmax(predicted_prob,1),prob)
                        
            loss=lambada*mseloss(predicted_emb,emb)/batch_size + gamma*loss_audioset

            all_y.append(emb.cpu().numpy().reshape(batch_size,-1))
            all_y_predicted.append(predicted_emb.detach().cpu().numpy().reshape(batch_size,-1))
            running_loss += loss.item()
            

        r2_model = r2_score(np.vstack(all_y),np.vstack(all_y_predicted),multioutput='raw_values') 
        return running_loss/batch_nb, r2_model