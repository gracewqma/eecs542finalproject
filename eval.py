import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.utils as vutils

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

import argparse
import os
from model import *
from PIL import Image

def inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def generate_images(generator, args, num_imgs=1, noise=None, ):
    """Generate images from the generator

    generator -- the generator model
    num_imgs -- number of images to generate
    noise -- optional noise to put in generator
    """
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    if noise is None:
        noise = torch.randn(num_imgs, args.nz, 1, 1, device=device)

    # if torch.cuda.is_available():
    #     noise = Variable(torch.Tensor(noise).cuda())
    # else:
    #     noise = Variable(torch.Tensor(noise))

    gen_imgs = generator(noise).cpu()
    # gen_imgs = gen_imgs.data.cpu().numpy()

    return gen_imgs




if __name__ == '__main__':
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/root/logs', help='root directory')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3, help='number of channels')
    parser.add_argument('--nz', type=int, default=100, help='dimension of latent')
    parser.add_argument('--ngf', default=64, type=int,help='size of feature maps in generator')
    parser.add_argument('--ndf', default=64, type=int,help='size of feature maps in discriminator')
    parser.add_argument('--num_epochs', default=1000, type=int,help='number of training epochs')
    parser.add_argument('--lr', default=0.0002, type=float,help='learning rate for optimizers')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1 hyperparam for Adam optimizers')
    parser.add_argument('--ngpu', default=1, type=int, help='number of GPUs available, 0 for CPU')

    args = parser.parse_args()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    G_path = os.path.join(args.root_dir, args.exp_name, 'netG_recent.pth')

    # load model checkpoint
    netG = Generator(args).to(device)
    if (device.type == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    netG.load_state_dict(torch.load(G_path))
    netG.eval()

    # generate images
    num_imgs = 30
    gen_imgs = generate_images(netG, args, num_imgs=num_imgs, noise=None)

    # save images
    imgs_basedir = os.path.join(args.root_dir, args.exp_name, 'generated')
    imgs_dir = os.path.join(imgs_basedir, 'imgs')

    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    
    print(gen_imgs.shape)
    
    for i in range(num_imgs):
        vutils.save_image(gen_imgs[i], os.path.join(imgs_dir, '{}.png'.format(i)), normalize=True)
    
        # img = gen_imgs[i]
        # img = img * 0.5 + 0.5
        # img = img.transpose(1, 2, 0)
        # img = Image.fromarray(img)
        # img.save(os.path.join(imgs_dir, '{}.png'.format(i)))
        # break
        



    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    image_dataset = dset.ImageFolder(root=imgs_basedir,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))


    IgnoreLabelDataset(image_dataset)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(image_dataset), cuda=True, batch_size=8, resize=True, splits=10))