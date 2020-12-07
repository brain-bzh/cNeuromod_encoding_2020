import torch, warnings
import torch.nn as nn
from nistats import hemodynamic_models
import numpy as np
from models import soundnet_model as snd

class SoundNetEncoding(nn.Module):
    def __init__(self,pytorch_param_path,nroi=210,fmrihidden=1000,nroi_attention=None, hrf_model=None, oversampling = 16, tr = 1.49, audiopad = 0):
        super(SoundNetEncoding, self).__init__()

        self.soundnet = snd.SoundNet8_pytorch()
        self.fmrihidden = fmrihidden
        self.nroi = nroi

        print("Loading SoundNet weights...")
        # load pretrained weights of original soundnet model
        self.soundnet.load_state_dict(torch.load(pytorch_param_path))

        #freeze the parameters of soundNet
        for param in self.soundnet.parameters():
            param.requires_grad = False

        print("Pretrained model loaded")

        self.gpool = nn.AdaptiveAvgPool2d((1,1)) # Global average pooling

        self.encoding_fmri = nn.Sequential(                
                nn.Linear(1024,self.fmrihidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.fmrihidden,self.nroi)
            )
            
        if hrf_model is not None : 
            self.hrf_model = hrf_model
            self.oversampling = oversampling
            self.audiopad = audiopad
            self.tr = tr
        else :
            self.hrf_model=None

    def forward(self, x, onsets, durations):
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            emb = self.soundnet(x)
            emb = self.gpool(emb)
            emb = emb.view(-1,1024)

            if self.hrf_model is not None :
                fvs = emb.cpu().numpy()

                index_zeros = np.where(onsets == 0)
                if len(index_zeros[0]) > 0 and index_zeros[0][0] != 0:
                    n_onsets = np.split(onsets, index_zeros[0]) 
                    n_durations = np.split(durations, index_zeros[0])
                    fvs = np.split(fvs, index_zeros[0])
                else:
                    n_onsets = [onsets] 
                    n_durations = [durations]
                    fvs = [fvs]

                all_fv = np.array([]).reshape(emb.shape[1], 0)
                for onset, duration, fv in zip(n_onsets, n_durations, fvs):
                    fv_temp = []
                    frame_times = onset.numpy()

                    for amplitude in fv.T:
                        exp_conditions = (onset, duration, amplitude)
                        signal, _ = hemodynamic_models.compute_regressor(exp_conditions, 
                                    self.hrf_model, frame_times, oversampling=self.oversampling, min_onset=0)
                        fv_temp.append(signal)
                    b = np.squeeze(np.stack(fv_temp))
                    all_fv = np.concatenate((all_fv, b),axis=1)

                emb = np.squeeze(np.stack(all_fv)).T
                emb = torch.from_numpy(emb).float().cuda()

        out = self.encoding_fmri(emb)
        
        return out

class SoundNetEncoding_conv(nn.Module):
    def __init__(self,pytorch_param_path,nroi=210,fmrihidden=1000,nroi_attention=None, hrf_model=None, oversampling = 16, tr = 1.49, audiopad = 0,transfer=True,preload=True):
        super(SoundNetEncoding_conv, self).__init__()

        self.soundnet = snd.SoundNet8_pytorch()
        self.fmrihidden = fmrihidden
        self.nroi = nroi

        if preload:
            print("Loading SoundNet weights...")
            # load pretrained weights of original soundnet model
            self.soundnet.load_state_dict(torch.load(pytorch_param_path))
            print("Pretrained model loaded")
            if transfer:
                #freeze the parameters of soundNet
                print("Transfer learning - backbone is fixed")
                for param in self.soundnet.parameters():
                    param.requires_grad = False
            else:
                print("Finetuning : backbone will be optimized")

        self.encoding_fmri = nn.Sequential(                
                nn.Conv2d(1024,self.nroi,kernel_size=(1,1)),
                #nn.ReLU(inplace=True),
                #nn.Conv2d(self.fmrihidden,self.nroi,kernel_size=(1,1)),

            )

        if nroi_attention is not None:
            self.maskattention = torch.nn.Parameter(torch.rand(nroi,nroi_attention))
        else:
            self.maskattention = None
        
            
        if hrf_model is not None : 
            self.hrf_model = hrf_model
            self.oversampling = oversampling
            self.audiopad = audiopad
            self.tr = tr
        else :
            self.hrf_model=None

    def forward(self, x):
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            emb = self.soundnet(x)

        out = self.encoding_fmri(emb)
        
        return out


class SoundNetEncoding_conv_2(nn.Module):
    def __init__(self,pytorch_param_path,nroi=210,fmrihidden=1000,nroi_attention=None, hrf_model=None, oversampling = 16, tr = 1.49, audiopad = 0,transfer=True,preload=True):
        super(SoundNetEncoding_conv_2, self).__init__()

        self.soundnet = snd.SoundNet8_pytorch()
        self.fmrihidden = fmrihidden
        self.nroi = nroi

        if preload:
            print("Loading SoundNet weights...")
            # load pretrained weights of original soundnet model
            self.soundnet.load_state_dict(torch.load(pytorch_param_path))
            print("Pretrained model loaded")
            if transfer:
                #freeze the parameters of soundNet
                print("Transfer learning - backbone is fixed")
                for param in self.soundnet.parameters():
                    param.requires_grad = False
            else:
                print("Finetuning : backbone will be optimized")

        self.encoding_fmri = nn.Sequential(                
                nn.Conv2d(1024,self.fmrihidden,kernel_size=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.fmrihidden,self.nroi,kernel_size=(1,1)),

            )

        if nroi_attention is not None:
            self.maskattention = torch.nn.Parameter(torch.rand(nroi,nroi_attention))
        else:
            self.maskattention = None
        
            
        if hrf_model is not None : 
            self.hrf_model = hrf_model
            self.oversampling = oversampling
            self.audiopad = audiopad
            self.tr = tr
        else :
            self.hrf_model=None

    def forward(self, x):
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            emb = self.soundnet(x)

        out = self.encoding_fmri(emb)
        
        return out


class SoundNetEncoding_conv_3(nn.Module):
    def __init__(self,pytorch_param_path,nroi=210,fmrihidden=1000,nroi_attention=None, hrf_model=None, oversampling = 16, tr = 1.49, audiopad = 0,transfer=True,preload=True):
        super(SoundNetEncoding_conv_3, self).__init__()

        self.soundnet = snd.SoundNet8_pytorch()
        self.fmrihidden = fmrihidden
        self.nroi = nroi

        if preload:
            print("Loading SoundNet weights...")
            # load pretrained weights of original soundnet model
            self.soundnet.load_state_dict(torch.load(pytorch_param_path))
            print("Pretrained model loaded")
            if transfer:
                #freeze the parameters of soundNet
                print("Transfer learning - backbone is fixed")
                for param in self.soundnet.parameters():
                    param.requires_grad = False
            else:
                print("Finetuning : backbone will be optimized")

            

        self.encoding_fmri = nn.Sequential(                
                nn.Conv2d(1024,2*self.fmrihidden,kernel_size=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(2*self.fmrihidden,self.fmrihidden,kernel_size=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.fmrihidden,self.nroi,kernel_size=(1,1)),

            )

        if nroi_attention is not None:
            self.maskattention = torch.nn.Parameter(torch.rand(nroi,nroi_attention))
        else:
            self.maskattention = None
        
            
        if hrf_model is not None : 
            self.hrf_model = hrf_model
            self.oversampling = oversampling
            self.audiopad = audiopad
            self.tr = tr
        else :
            self.hrf_model=None

    def forward(self, x):
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            emb = self.soundnet(x)

        out = self.encoding_fmri(emb)
        
        return out