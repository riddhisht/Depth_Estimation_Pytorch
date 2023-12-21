import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F 


activations = {}
"""
Inside the hook function, the line activations[name] = output records 
the output of the module (the intermediate activation) in the 
activations dictionary with the key name.

Finally, the get_activation function returns the hook function. This
 allows you to create a hook for a specific layer of your neural 
 network and assign it a name. When this hook is registered with a 
 PyTorch module using the register_forward_hook method, it will 
 capture the intermediate activation of that module during the 
 forward pass and store it in the activations dictionary under the 
 specified name.
"""
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

attention = {}

def get_attention(name):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape

        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2,0,3,1,4)
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        ) # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2,-1)) * module.scale
        attention[name] = attn

    return hook

def get_mean_attention_map(attn, token, shape):
    attn = attn[:,:, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn

class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]
    
class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index ==2:
            readout = (x[:,0] + x[:,1]) / 2

        else:
            readout = x[:, 0]

        return x[:, self.start_index :] + readout.unsqueeze(1)
        
class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2*in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:,0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)
    
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x
    
    