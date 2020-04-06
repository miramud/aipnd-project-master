import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import n_utitls

ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, type=float)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5, type=float)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs


trainloader, v_loader, testloader = n_utitls.load_data(where)

model, optimizer, criterion = n_utitls.network_setup(structure,dropout,hidden_layer1,lr)

if torch.cuda.is_available() and power =='gpu':
        model.to('cuda:0')
n_utitls.network_function(model, optimizer, criterion, epochs, 40, trainloader, power)

n_utitls.save_check_point(path,structure,hidden_layer1,dropout,lr)

print("Hurrayyyyyyy, Training Done!!!")