# EECS 542 Final Project - Extending Differentiable Augmentations to Style Transfer

augmentations.py include all of the augmentations that we implemented for DCGAN

model.py is our DCGAN model

train.py is how we trained our DCGAN model with augmentations

FID comaprisons was ran using pytorch-FID

cut/models/aug_class.py was all the augmentation that we implemented for CUTGAN, including saving random variables to ensure the same augmentations were applied between calls.

cut/models/cut_model.py is our modified CUT-GAN model with our differentiable augmentations. 

cut/train.py is how we trained out CUT-GAN model
