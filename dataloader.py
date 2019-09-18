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


class PersonDataset(Dataset):

    def __init__(self, dataset_dir ):
        """

        Args:
            dataset_dir : path to the dataset
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.dataset_dir = dataset_dir
        self.list_dir = os.listdir(self.dataset_dir)




    def __len__(self):
        return len(self.list_dir)


    def transform(self,image,masks):
        # convert to PIL Image.
        PIL_convert = transforms.ToPILImage()
        image = PIL_convert(image)
        masks = PIL_convert(masks.astype(np.int32))
        # resize
        resize = transforms.Resize(size=(512,512))
        image = resize(image)
        masks = resize(masks)

        '''
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            masks = TF.hflip(masks)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            masks = TF.vflip(masks)
        '''
        # Convert to Tensor
        image = TF.to_tensor(image)
        masks = TF.to_tensor(masks)

        return image,masks

    def __getitem__(self,image_id):

        # read the image
        image_path  = (os.path.join(self.dataset_dir,self.list_dir[image_id],"images/{}.png".format(self.list_dir[image_id])))
        image = io.imread(image_path)
        # read the mask
        mask_dir = os.path.join(self.dataset_dir,self.list_dir[image_id],'masks')
        masks_list = []

        for i, f in enumerate (next(os.walk(mask_dir))[2]):
            if f.endswith ('.png'):
                m = io.imread(os.path.join(mask_dir,f)).astype(np.bool)
                m = m[:,:,0]
                masks_list.append(m)

                if len(masks_list) != 1:
                    masks = np.logical_or(masks,masks_list[i])
                else:
                    masks = masks_list[i]
        # do the transforms..
        trans_img,trans_masks = self.transform(image,masks)
        sample = {"image":trans_img,"masks":trans_masks}

        return(sample)
def get_dataloaders(data_dir,train_batch_size,val_batch_size):
    # using pytorch SUBSETRANDOMSAMPLER

    #data directory
    #data_dir =  data_dir

    # Create the dataset object.
    transformed_dataset  = PersonDataset(data_dir)
    # dataloader for train and validation
    validation_split = 0.2
    shuffle_dataset = True
    random_seed= 42
    # create indices for training and validation splits.
    dataset_size  = len(transformed_dataset)
    # we create the indices using python range function and store it into a list
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split*dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices,val_indices = indices[split:],indices[:split]
    # create dataloaders...
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler   = SubsetRandomSampler(val_indices)

    train_loader  = DataLoader(transformed_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0,sampler = train_sampler)
    val_loader  = DataLoader(transformed_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0,sampler = val_sampler)

    # dictionary for data loaders..
    dataloaders = {"train" :train_loader,
                "val":val_loader
                }
    return dataloaders
