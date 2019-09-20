import os
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import copy
import time
import argparse
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
from model import Unet
from utils import VisdomLinePlotter,train_val


def read_flags():
    """Return Global variables"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--dataset_dir",
    default = '/home/robin/Thesis/Dataset/ds12',
    help="path/to/root_directory/Dataset")

    parser.add_argument(
    "--log_dir",
    default = '/home/robin/repos/logs/',
    help = "path_to/log_directory/")

    parser.add_argument(
    "--aug",
    default = False,
    help = "flag to define whether augmentation is needed or not")

    parser.add_argument(
    "--train_batch_size",
    default = 1,
    help = "define the training batch size")

    parser.add_argument(
    "--val_batch_size",
    default = 1,
    help = "define the validation batch size")

    parser.add_argument(
    "--epochs",
    default = 100,
    help = "define the number of epochs for training")

    return parser.parse_args()

def main(FLAGS):

    "train and validate the Unet model"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #data directory
    data_dir =  FLAGS.dataset_dir
    #log_directory
    log_dir = FLAGS.log_dir
    # Hyper and other parameters
    train_batch_size = FLAGS.train_batch_size
    val_batch_size   = FLAGS.val_batch_size
    aug_flag = FLAGS.aug
    num_epochs = FLAGS.epochs
    num_classes = 2
    # get the train and validation dataloaders
    dataloaders = get_dataloaders(data_dir,train_batch_size,val_batch_size,aug_flag)
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
    train_val(dataloaders,model,criterion,optimizer,num_epochs,log_dir,device)

if __name__ == '__main__':
    flags = read_flags()
    main(flags)
