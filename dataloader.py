import os
from skimage import io, transform
from skimage import img_as_bool
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import SubsetRandomSampler



class PersonDataset(Dataset):
    """ Peerson Dataset. """

    def __init__(self, dataset_dir,aug ):
        """
        Args:
            dataset_dir(str) : path to the dataset(root dir) and arranged as follows
                             ├── Dataset
                             │   ├── sample.png
                             │      ├──images
                             │        ├── sample.png
                             │      ├── masks
                             │        ├── id[0].png
                                      └── id[i].png
            aug(boolean): variable to determine the need of augmentation
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.dataset_dir = dataset_dir
        self.list_dir = os.listdir(self.dataset_dir)
        self.aug = aug



    def __len__(self):
        """To return the length of dataset"""
        return len(self.list_dir)

    def augment(image,masks):
        """
        Applying the same augmentation to
        image and its corresponding mask
        Args:
            image(PIL Image): resized image(new_width,new_height)
            masks(PIL Image): resized mask(new_width,new_height)
        """

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            masks = TF.hflip(masks)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            masks = TF.vflip(masks)
        return image,masks

    def transform(self,image,masks,aug):
        """Applying a set of  transformations as a datapreprocessing task"""
        # convert to PIL Image.
        PIL_convert = transforms.ToPILImage()
        image = PIL_convert(image)
        masks = PIL_convert(masks.astype(np.int32))
        # resize the image and masks
        resize = transforms.Resize(size=(512,512))
        image = resize(image)
        masks = resize(masks)
        # augmentation
        if aug is True:
            augment(image,masks)
        else:
            pass
        # Convert to Tensor
        image = TF.to_tensor(image)
        masks = TF.to_tensor(masks)

        return image,masks

    def __getitem__(self,image_id):
        """
        Function to read the image and mask
        and return a sample of dataset when neededself.
        Args:
        image_id: image index to iterate over the dataset samples
        Returns:
        sample(dict): a sample of the dataset

        """
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
                #combine all the masks corresponding of an invidual sample image into single binary mask
                if len(masks_list) != 1:
                    masks = np.logical_or(masks,masks_list[i])
                else:
                    masks = masks_list[i]
        # do the transforms..
        trans_img,trans_masks = self.transform(image,masks,self.aug)
        sample = {"image":trans_img,"masks":trans_masks}

        return(sample)
def get_dataloaders(data_dir,train_batch_size,val_batch_size,aug_flag):
    """
    Function to create train  and validation dataloaders
    Pytorch SUBSETRANDOMSAMPLER to create the train-val split

    Args:
    data_dir(str):root directory of the dataset_dir
    train_batch_size(int):Mini batch size for training
    val_batch_size(int):Mini batch size for validation

    Returns:
    dataloaders(dict): Dictionary of dataloaders
    """
    # Create the dataset object.
    transformed_dataset  = PersonDataset(data_dir,False)
    # dataloader for train and validation
    validation_split = 0.2
    shuffle_dataset = True
    #random seed to keep the train-val split constant for inference purpose
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
    train_aug,val_aug = aug_flag,False
    train_loader  = DataLoader(PersonDataset(data_dir,train_aug), batch_size=train_batch_size, shuffle=False, num_workers=0,sampler = train_sampler)
    val_loader  = DataLoader(PersonDataset(data_dir,val_aug), batch_size=val_batch_size, shuffle=False, num_workers=0,sampler = val_sampler)

    # dictionary for data loaders..
    dataloaders = {"train" :train_loader,
                "val":val_loader
                }
    return dataloaders
