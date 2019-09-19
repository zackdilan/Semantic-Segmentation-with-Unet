import os
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import copy
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from visdom import Visdom

from dataloader import get_dataloaders
from network_1024 import Unet
from utils import VisdomLinePlotter



if __name__ == "__main__":

    "train and validate the Unet model"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #data directory
    data_dir =  '/home/chacko/data/supervisely_person_data'
    # Hyper and other parameters
    train_batch_size = 12
    val_batch_size   = 1
    num_epochs = 100
    learning_rate = 0.01            #
    num_classes = 2
    # get the train and validation dataloaders
    dataloaders = get_dataloaders(data_dir,train_batch_size,val_batch_size)
    model = Unet(3,num_classes)

    # Uncomment to run traiing on Multiple GPUs
    if torch.cuda.device_count()>1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model,device_ids = [0,1])
    else:
        print("no multiple gpu found")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(),
                          lr=0.02,
                          momentum=0.9,
                          weight_decay=0.0005)


    #optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size = 10,gamma = 0.1)
    plotter = VisdomLinePlotter(env_name='Unet Train')
    # uncomment for leraning rate schgeduler..
    #train_val(dataloaders,model,criterion,optimizer,exp_lr_scheduler,num_epochs)
    train_val(dataloaders,model,criterion,optimizer,num_epochs)
