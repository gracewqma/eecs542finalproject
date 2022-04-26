# EECS 542 Final Project - Extending Differentiable Augmentations to Style Transfer

## File Descriptions

**augmentations.py**: augmentation implementations for DCGAN

**model.py**: DCGAN model implementation

**train.py**: training code for DCGAN model with augmentations

FID comparisons was ran using **pytorch-FID**

**cut/models/aug_class.py**: augmentation implementations for CUTGAN, including saving random variables to ensure the same augmentations were applied between calls.

**cut/models/cut_model.py**: modified CUT-GAN model with our differentiable augmentations. 

**cut/train.py**: training code for CUT-GAN model
