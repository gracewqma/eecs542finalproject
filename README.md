# EECS 542 Final Project - Extending Differentiable Augmentations to Style Transfer
## Description ##
We implement differentiable augmentations with respect to two networks: DCGAN and CUT-GAN. We implement color, translation, cutout transforms from the authors, as well as our own with hflip, rotation, Gaussian blur, and Gaussian sharpen. Our code uses DCGAN implemented from Pytorch's tutorial, and we build on top of CUT-GANs repository. Both are cited below. To train the model, arguments can be specified in the corresponding config file and the command for DCGAN is:
```
./run_train.sh
```
and for CUT-GAN, the corresponding script is: 
```
./run_cut.sh
```
**Note:** Datasets are not included, and can be downloaded from the original paper repositories. 

## CUT-GAN With Differentiable Augment Results ##

<img src="https://github.com/gracewqma/eecs542finalproject/blob/main/translation.png">

## File Descriptions

**augmentations.py**: augmentation implementations for DCGAN

**model.py**: DCGAN model implementation

**train.py**: training code for DCGAN model with augmentations

FID comparisons was ran using **pytorch-FID**

**cut/models/aug_class.py**: augmentation implementations for CUTGAN, including saving random variables to ensure the same augmentations were applied between calls.

**cut/models/cut_model.py**: modified CUT-GAN model with our differentiable augmentations. 

**cut/train.py**: training code for CUT-GAN model


### Citations ###
```
@article{fix,
  title={fix},
  author={fix},
  year={fix}
}
```
