import torch, warnings
import torch.nn as nn
from nistats import hemodynamic_models
import numpy as np

#adapted from https://github.com/smallflyingpig/SoundNet_Pytorch

class SoundNet8_pytorch(nn.Module):
    def __init__(self):
        super(SoundNet8_pytorch, self).__init__()
        
        self.define_module()
        self.indexes= {"conv1" : 0, "pool1" : 1, "conv2" : 2, "pool2" : 3, "conv3" : 4, "conv4" : 5, 
                        "conv5" : 6, "pool5" : 7, "conv6" : 8, "conv7" : 9, "conv8" : 10, "conv8_2" : 11} 
        self.layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.conv4, 
                        self.conv5, self.pool5, self.conv6, self.conv7, self.conv8, self.conv8_2]
        self.layers_size = {"conv1" : 16, "pool1" : 16, "conv2" : 32, "pool2" : 32, "conv3" : 64, "conv4" : 128,
                            "conv5" : 256, "pool5" : 256, "conv6" : 512, "conv7" : 1024, "conv8" : 1000, "conv8_2" : 401} 
        
    def define_module(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16, (64,1), (2,1), (32,0), bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d((8,1),(8,1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (32,1), (2,1), (16,0), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d((8,1),(8,1))

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
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d((4,1),(4,1))

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

    #-----------------------WIP----------------
    def forward(self, x, output_layer = "conv8", train_start = None):
        warnings.filterwarnings("ignore")

        # self.indexes= {"conv1" : 0, "pool1" : 1, "conv2" : 2, "pool2" : 3, "conv3" : 4, "conv4" : 5, 
        #                 "conv5" : 6, "pool5" : 7, "conv6" : 8, "conv7" : 9, "conv8" : 10, "conv8_2" : 11} 
        # self.layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.conv4, 
        #                 self.conv5, self.pool5, self.conv6, self.conv7, self.conv8, self.conv8_2]
        # self.layers_size = {"conv1" : 16, "pool1" : 16, "conv2" : 32, "pool2" : 32, "conv3" : 64, "conv4" : 128,
        #                     "conv5" : 256, "pool5" : 256, "conv6" : 512, "conv7" : 1024, "conv8" : 1000, "conv8_2" : 401} 

        output_index =  self.indexes[output_layer] if output_layer != "conv8"  else 10
        train_start_index = self.indexes[train_start] if train_start != None else 
        #train_start_index is the index of the FIRST layer to be trained, if train_start == None --> transfer learning without finetuning
 
        with torch.no_grad(): 
            for net in self.layers[:train_start_index]:
                x = net(x) 
        for net in self.layers[train_start_index:output_index]:
            x = net(x) 

        if output_layer == "conv8":
            object_pred = self.conv8(x)
            scene_pred = self.conv8_2(x)
            return (object_pred, scene_pred)
        else :  
            return x
    #----------------------END-WIP----------------

    def extract_feat(self,x:torch.Tensor)->list:
        output_list = []
        for net in [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, 
                    self.conv4, self.conv5, self.pool5, self.conv6, self.conv7]:
            x = net(x)
            output_list.append(x.detach().cpu().numpy())
        object_pred = self.conv8(x)
        output_list.append(object_pred.detach().cpu().numpy())
        scene_pred = self.conv8_2(x) 
        output_list.append(scene_pred.detach().cpu().numpy())
        return output_list