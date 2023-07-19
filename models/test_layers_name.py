
import torch, warnings
import torch.nn as nn
import soundnet_model as snd

pytorch_param_path='./sound8.pth'

output_layer=7
train_limit=5
soundnet = snd.SoundNet8_pytorch(output_layer, train_limit)
print("Loading SoundNet weights...")
# load pretrained weights of original soundnet model
soundnet.load_state_dict(torch.load(pytorch_param_path))
print("Pretrained model loaded")

#freeze the parameters of soundNet
print("Transfer learning - backbone is fixed")
for layer, param in enumerate(soundnet.parameters()):
    if layer < (train_limit-1)*4 :
        param.requires_grad = False
    else : 
        param.requires_grad = True

for i, (name, param) in enumerate(soundnet.named_parameters()):
    print(name, param.requires_grad)

