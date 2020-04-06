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

sets = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

def transform_image(root):
        
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms ={
    'train_transforms':transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),

    'test_transforms' : transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),

    'validation_transforms' : transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])}

    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train_data' : datasets.ImageFolder(train_dir, transform = data_transforms['train_transforms']),
    'test_data' : datasets.ImageFolder(test_dir ,transform = data_transforms['test_transforms']),
    'valid_data' : datasets.ImageFolder(valid_dir, transform = data_transforms['validation_transforms'])
                    }

    return image_datasets['train_data'] , image_datasets['valid_data'], image_datasets['test_data']    

def load_data(root):
    
    data_dir = root    
    tr_data,val_data,te_data=transform_image(data_dir)

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    dataloaders = {'train_data_loader' : torch.utils.data.DataLoader(tr_data, batch_size=64, shuffle=True),
    'test_data_loader' : torch.utils.data.DataLoader(te_data, batch_size=32),
    'valid_data_loader' : torch.utils.data.DataLoader(val_data, batch_size=32)}
    
    return dataloaders['train_data_loader'] , dataloaders['valid_data_loader'], dataloaders['test_data_loader']


train_data,valid_data,test_data=transform_image('./flowers/')
trdl,vdl,tsdl=load_data('./flowers/')


def network_setup(structure='vgg16',dropout=0.5, hidden_layer1 = 4096,lr = 0.001):
    
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} is not a valid model. Please try either vgg16, densenet121, or alexnet?".format(structure))
        
    
        
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(sets[structure],hidden_layer1)),
                        ('relu1', nn.ReLU()),
                        ('d_out1',nn.Dropout(dropout)),
                        ('fc2', nn.Linear(hidden_layer1, 1024)),
                        ('relu2', nn.ReLU()),
                        ('d_out2',nn.Dropout(dropout)),
                        ('fc3', nn.Linear(1024, len(tsdl))),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))
    
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )
    model.cuda()
    
    return model , optimizer ,criterion 

# model,optimizer,criterion = network_setup('vgg16')


# TODO: Build and train your network

def network_function(model, train_data_loader, epochs, print_every, criterion, optimizer, device='cpu'):
    
    print_every = print_every
    steps = 0
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_data_loader):
            steps += 1
            if torch.cuda.is_available() and device =='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_lost = 0
                valid_accuracy=0


                for ii, (inputs2,labels2) in enumerate(vdl):
                    optimizer.zero_grad()

                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda')
                        model.to('cuda')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        valid_lost = criterion(outputs,labels2)
                        passes = torch.exp(outputs).data
                        equality = (labels2.data == passes.max(1)[1])
                        valid_accuracy += equality.type_as(torch.FloatTensor()).mean()

                valid_lost = valid_lost / len(vdl)
                valid_accuracy = valid_accuracy /len(vdl)



                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(valid_lost),
                       "Accuracy: {:.4f}".format(valid_accuracy))


                running_loss = 0
    print('*** Training Successfully Completed ****')
    print('Epochs: ', epochs)
    print('Steps: ', steps)
# network_function(model, dataloaders['train_data_loader'], 3, 40, criterion, optimizer, 'gpu')

# Save the checkpoint
def save_check_point(model=0,path='checkpoint.pth',structure ='vgg16', hidden_layer1 = 4096,dropout=0.5,lr=0.001,epochs=3): 
    model.class_to_idx =  train_data.class_to_idx
    model.cpu()
    torch.save({'structure' :structure,
                'hidden_layer1': hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
# Load the checkpoint
def load_check_point(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    lr=checkpoint['lr']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    structure = checkpoint['structure']
    model,_,_ = network_setup(structure , dropout,hidden_layer1,lr)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
# model=load_check_point('checkpoint.pth')

def process_image(image='/home/workspace/aipnd-project-master/flowers/test/1/image_06752.jpg'):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pro_img = Image.open(image)
   
    transform_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    # TODO: Process a PIL image for use in a PyTorch model
    img_py = transform_img(pro_img)
    return img_py

def predict(image='/home/workspace/aipnd-project-master/flowers/test/1/image_06752.jpg', model=0, topk=5, device='gpu'):
    if torch.cuda.is_available() and device =='gpu':
        model.to('cuda:0')
    image_torch = process_image(image)
    image_torch = image_torch.unsqueeze_(0)
    image_torch = image_torch.float()
    
    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(image_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(image_torch)
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)