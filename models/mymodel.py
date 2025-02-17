import convnet
import torch.nn as nn
import torch.nn.functional as F

def make_model(args):
    return mymodel(args)

class mymodel(nn.Module):
