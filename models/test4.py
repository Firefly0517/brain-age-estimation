import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from torch_geometric.nn import TransformerConv, GraphConv, GCN2Conv, GCNConv, DenseGraphConv
from torch.utils.data import random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from observer import Runtime_Observer
from utils_mamba import MambaLayer, Mamba_node



def make_model(args):
    return graph_model(args,
                      )

class graph_model(nn.Module):
    def __init__(self, args):
        super(graph_model, self).__init__()
        self.args = args




