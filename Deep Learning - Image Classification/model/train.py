#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import  vgg16
from torch import nn, optim

import torch.nn.functional as F
from collections import OrderedDict
import torchvision

import json
import seaborn as sns
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Training process to identify flowers")
    parser.add_argument('--data_dir', default='flowers_data', type=str, help='set the data dir')
    parser.add_argument('--arch', dest='arch', default='vgg16', type=str, choices=['vgg16', 'vgg19'], help='choose a pretrained model')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001, help='choose a learning rate' )
    parser.add_argument('--hidden_units', dest='hidden_units', type= list, default=[4096,1024], help='set hidden units')
    parser.add_argument('--epochs', dest='epochs', type=int, default=8, help='set number of epochs')
    parser.add_argument('--gpu', dest='gpu', type=bool, default=True , help='whether to turn on GPU')
    return parser.parse_args()

def validation(model, data_loader, criterion,gpu):
    test_loss = 0
    accuracy = 0
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()
    for inputs, labels in data_loader:
        if cuda :
            inputs,labels = inputs.to('cuda'), labels.to('cuda')
        output = model(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)


def train(model, criterion, optimizer, train_dataloader, valid_dataloader, epochs, gpu):

    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()
    steps=0
    print_every=60
    for e in range(epochs):
        running_loss = 0
        accuracy = 0
        model.train()
        for inputs, labels in train_dataloader:

            steps += 1 
            if cuda :
                inputs,labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1]) # check if prediction matches the actual label
            accuracy += equality.type(torch.FloatTensor).mean()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    val_loss, val_accuracy = validation(model, valid_dataloader, criterion,gpu)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Training Accuracy: {:.3f}".format(accuracy/print_every),
                      "Validation Loss: {:.3f}.. ".format(val_loss),
                      "Validation Accuracy: {:.3f}".format(val_accuracy))

                # clear out accumulated loss , accuracy
                running_loss = 0
                accuracy = 0
                # Make sure training is back on
                model.train()

def build_classifier(num_in_features, hidden_layers, num_out_features):
    """Build a fully connected network
    hidden_layers: None or a list, e.g. [512, 256, 128]
    """

    classifier = nn.Sequential()
    if hidden_layers == None:
        classifier.add_module('fc0', nn.Linear(num_in_features, num_out_features))
    else:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
        classifier.add_module('relu0', nn.ReLU())
        classifier.add_module('drop0', nn.Dropout(.2))
        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
            classifier.add_module('relu'+str(i+1), nn.ReLU())
            classifier.add_module('drop'+str(i+1), nn.Dropout(.2))
        classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))
        classifier.add_module('softmax', nn.LogSoftmax(dim=1))
    return classifier

def save_checkpoint (model, classifier, optimizer, epochs):
    args = parse_args()
    #model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {
    'epoch': epochs,
    'arch' : args.arch,
    'model': model,
    'state_dict': model.state_dict(),
    'classifier': classifier,
    'optimizer_dict': optimizer.state_dict(), 
    'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, 'checkpoint.pth')

def main ():
    args = parse_args()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_datasets= datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(validation_datasets, batch_size=64, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)

    model = getattr(models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

        # Hyperparameters for our network
    input_size = model.classifier[0].in_features 
    output_size = 102
    hidden_layers = args.hidden_units
    # Build a feed-forward network
    classifier = build_classifier(input_size, hidden_layers, output_size)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    epochs = args.epochs
    gpu = args.gpu
    train(model, criterion, optimizer, train_dataloaders, valid_dataloaders, epochs, gpu)

    # save the checkpoint
    model.class_to_idx = train_datasets.class_to_idx
    save_checkpoint(model, classifier, optimizer, epochs)

if __name__ == '__main__':
    main()