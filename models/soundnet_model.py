import torch, warnings
import torch.nn as nn
from nistats import hemodynamic_models
import numpy as np

class SoundNet8_pytorch(nn.Module):
    def __init__(self):
        super(SoundNet8_pytorch, self).__init__()
        
        self.define_module()
        
    def define_module(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16, (64,1), (2,1), (32,0), bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8,1), (8,1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (32,1), (2,1), (16,0), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8,1),(8,1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (16,1), (2,1), (8,0), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (8,1), (2,1), (4,0), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, (4,1),(2,1),(2,0), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,1),(4,1))
        ) # difference here (0.24751323, 0.2474), padding error has beed debuged
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, (4,1), (2,1), (2,0), bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, (4,1), (2,1), (2,0), bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(1024, 1000, (8,1), (2,1), (0,0), bias=True),
        ) 
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(1024, 401, (8,1), (2,1), (0,0), bias=True)
        )

    def forward(self, x):
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]:
            x = net(x)
            
        ### "Truncated" soundnet to only extract conv7 

        #object_pred = self.conv8(x)
        #scene_pred = self.conv8_2(x) 
        return x