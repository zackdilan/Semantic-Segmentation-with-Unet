# Semantic-Segmentation-with-Unet
Pytorch implementation of Human(person) segmentation from RGB images.

## Dataset
Dataset is available in https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets
Please  make sure to arrange the Dataset tree as follows.
dataset_dir(str) : path to the dataset(root dir) and arranged as follows.  

```bash
├── Dataset
│   ├── sample.png
│      ├──images
│        ├── sample.png
│      ├── masks
│        ├── id[0].png
         └── id[i].png
```
## Dependancies
1. Python3
2. Pytorch 1.1.0

## Unet Model Evaluation.
jupyter notebook(Unet_Evaluation.ipynb) for data loader  and model prediction is provided.

## Usage
python train.py  
If you want help - python train.py --help
