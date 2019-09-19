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



class VisdomLinePlotter(object):
    """
    To plot graphs while usin Pytorch
    for more info:https://github.com/noagarcia/visdom-tutorial
    """
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


def train_val(dataloaders,model,criterion,optimizer,num_epochs,log_dir,device):
    """
    training and validation function phase for each epoch
    Args:
    log_dir(str): mention the path to save the logfiles
    """
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
            correct_train  = 0
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

                #metric evaluation
                running_loss += loss.item()
                total_pixels += labels.nelement()  # to get the total number of pixels in the batch
                correct_pixels += predicted.eq(labels.data).sum().item()

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
                        },(os.path.join(log_dir,'train_exp-epoch{}.pth'.format(epoch))))
            else:
                pass

            time_elapsed = time.time()-since
            print("Epoch complete in {:.0f}min - {:.0f}secs".format(time_elapsed/60,time_elapsed%60))
