# Importing the necessary libraries
import argparse
import json
import time


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils import data

from collections import OrderedDict

from PIL import Image

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

def parse_args():
    # Defining the command line arguments
    parser = argparse.ArgumentParser(description='Train a network on a dataset of images')
    parser.add_argument('data_dir', type=str, default='./flowers', help='The directory where the dataset is stored')
    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str, default='checkpoint.pth', help='The directory where the model checkpoints are saved')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'], dest='arch', help='The architecture of the network')
    parser.add_argument('--hidden_units', type=int, dest='hidden_units', default=512, help='The number of hidden units in the classifier')
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.001, help='The learning rate of the optimizer')
    parser.add_argument('--epochs', type=int, default=2, dest='epochs', help='The number of epochs to train the network')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    args = parser.parse_args()
    return args


def load_data(path='./flowers'):
    print("Loading the data...........")
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
    }

    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle = True)
    
    # Get the number of classes from the dataset
    num_classes = len(train_data.classes)
    
    print("Data is loaded!")
    
    return train_data, trainloader, validloader, testloader, num_classes


def load_model(args, hidden_units, device, num_classes):
    print("Loading model.....")
    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    
    args = parse_args()

    if args.arch == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        num_of_features=25088
    else:
        model = torchvision.models.densenet121(pretrained=True)
        num_of_features=1024
    # Freeze the parameters of the feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace the classifier with a new one
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_of_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Move the model to the device
    model.to(device)
    # Define the loss function and the optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    print("Model:")
    print(model)
    
    return model, criterion, optimizer, num_of_features


 

def main():
    
    args = parse_args()
    train_data, trainloader, validloader, testloader, num_classes = load_data(path='./flowers')
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    model, criterion, optimizer, num_of_features = load_model(args, args.hidden_units, device, num_classes)
    
    

    
    
    start = time.time()
    print('Model is Training...')

    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(args.epochs):
        for inputs, labels in trainloader:
            steps += 1
          
            if torch.cuda.is_available() and args.gpu =='gpu':
                inputs, labels = inputs.to(device), labels.to(device)
                model = model.to(device)

            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                print("Validating......")
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

               
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
    print("Training is Finished")
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure' :args.arch,
                'hidden_units':args.hidden_units,
                'learning_rate':args.learning_rate,
                'no_of_epochs':args.epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    print("Saved checkpoint!")
    
    end = time.time()
    total_time = end - start
    print(" Model Trained in: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60)) 
if __name__ == "__main__":
    main()   

