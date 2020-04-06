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

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='./flowers/test/1/image_06752.jpg', action="store", type = str)
ap.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
ap.add_argument('checkpoint', default='/home/workspace/aipnd-project-master/checkpoint.pth', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
device = pa.gpu

path = pa.checkpoint

pa = ap.parse_args()

def main():
    model=n_utitls.load_check_point(path)
    with open(pa.category_names, 'r') as json_file:
        cat_to_name = json.load(json_file)
    probabilities = n_utitls.predict(path_image, model, number_of_outputs, device)
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    categories = [idx_to_class[idx] for idx in probabilities[1][0].tolist()]
    labels = [cat_to_name[cat] for cat in categories]
    probability = np.array(probabilities[0][0])
    i=0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Done Predicting!")

    
if __name__== "__main__":
    main()