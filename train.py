import os
from skimage import io, transform
from skimage import img_as_bool
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

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def train_val(dataloaders,model,criterion,optimizer,num_epochs):
    num_images = 0
    best_acc = 0

    for epoch in range(num_epochs):
        print("Epoch {}/{}". format(epoch,num_epochs-1))
        print("-"*50)
        since = time.time()
        running_loss = 0.0
        total_pixels = 0
        correct_pixels = 0
        # Each epoch has a training and validation phase
        for phase in ["train","val"]:
            if phase == "train":
                #scheduler.step()
                model.train()  ## set model in training mode
            else:
                model.eval()   ## set model in validation mode
            # define metric variables
            running_loss = 0.0
            total_train = 0
            correct_train = 0
            #iterate over the data
            for index,sampled_batch in enumerate(dataloaders[phase]):
                inputs = sampled_batch["image"].to(device)  # N,C,H,W
                labels = sampled_batch["masks"]  # N,C,H.W
                labels = torch.squeeze(labels,1).long().to(device) # N,H,W
                num_images += inputs.size(0)
                # zero the parameter gradients
                optimizer.zero_grad()
                #forward
                #trach history only in forward mode..
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _,predicted = torch.max(outputs,1)
                    loss = criterion(outputs,labels)

                    #backward + optimize only in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                #statistics
                running_loss += loss.item()
                total_pixels += labels.nelement()  # to get the total number of pixels in the batch
                correct_pixels += predicted.eq(labels.data).sum().item()
                '''
                if phase == "train":
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                        epoch, index, len(dataloaders[phase]), 100. * index / len(dataloaders[phase])))
                else:
                    pass
                '''
            # calculate the minibatch loss and accuracy and plot it using Visdom
            epoch_loss =  running_loss / num_images
            epoch_acc = 100 * correct_pixels / total_pixels
            if phase == 'train':
                plotter.plot('loss', 'train', 'Class Loss', epoch, epoch_loss)
                plotter.plot('accuracy', 'train', 'Class Acc', epoch, epoch_acc)
            else:
                plotter.plot('loss', 'validation', 'Class Loss', epoch, epoch_loss)
                plotter.plot('accuracy', 'validation', 'Class Acc', epoch, epoch_acc)
            print("{} --- Epoch {}, Epoch  loss : {:.4f} , Accuracy : {} %".format(phase ,epoch,epoch_loss,epoch_acc))
            # saving the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({ "epoch": epoch,
                            "model_state_dict":model.state_dict(),
                            "optimizer_state_dict":optimizer.state_dict(),
                            "loss":epoch_loss,
                        },'/home/chacko/data/logs/Exp_3/train_exp3-epoch{}.pth'.format(epoch))
            else:
                pass

            time_elapsed = time.time()-since
            print("Epoch complete in {:.0f}min - {:.0f}secs".format(time_elapsed/60,time_elapsed%60))

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
