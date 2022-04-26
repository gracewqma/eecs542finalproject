# EECS 542 Final Project - Extending Differentiable Augmentations to Style Transfer
## Description ##
We implement differentiable augmentations with respect to two networks: DCGAN and CUT-GAN. We implement color, translation, cutout transforms from the authors, as well as our own with hflip, rotation, Gaussian blur, and Gaussian sharpen. Our code uses DCGAN implemented from Pytorch's tutorial, and we build on top of CUT-GANs repository. Both are cited below.

The Docker image is in the Docker folder and can be built with:
```
cd docker
docker build . -t diffaugs
```

To train the model, arguments can be specified in the corresponding config file and the command for DCGAN is:
```
./run_train.sh
```
and for CUT-GAN, the corresponding script is: 
```
./run_cut.sh
```
Training plots can be visualized with tensorboard where specified in the ```log_dir``` argument.

**Note:** Datasets are not included, and can be downloaded from the original paper repositories. 

## CUT-GAN With Differentiable Augment Results ##

<img src="https://github.com/gracewqma/eecs542finalproject/blob/main/translation.png">

## File Descriptions

**augmentations.py**: augmentation implementations for DCGAN

**model.py**: DCGAN model implementation

**train.py**: training code for DCGAN model with augmentations

**split_datasets.py**: creates the 100-shot training subsample for the dataset.

FID comparisons was ran using **pytorch-FID**

**cut/models/aug_class.py**: augmentation implementations for CUTGAN, including saving random variables to ensure the same augmentations were applied between calls.

**cut/models/cut_model.py**: modified CUT-GAN model with our differentiable augmentations. 

**cut/train.py**: training code for CUT-GAN model


### Citations ###
```
@article{https://doi.org/10.48550/arxiv.2006.10738,
  title={Differentiable Augmentation for Data-Efficient GAN Training},
  author={Zhao, Shengyu and Liu, Zhijian and Lin, Ji and Zhu, Jun-Yan and Han, Song},
  year={2020}
}

@article{https://doi.org/10.48550/arxiv.2007.15651,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Park, Taesung and Efros, Alexei A. and Zhang, Richard and Zhu, Jun-Yan},
  year={2020}
}

@article{https://doi.org/10.48550/arxiv.1511.06434,
  title={Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks},
  author={Radford, Alec and Metz, Luke and Chintala, Soumith},
  year={2015}
}

@misc{Seitzer2020FID,
  author={Maximilian Seitzer},
  title={{pytorch-fid: FID Score for PyTorch}},
  month={August},
  year={2020},
  note={Version 0.2.1},
  howpublished={\url{https://github.com/mseitzer/pytorch-fid}},
}
```
