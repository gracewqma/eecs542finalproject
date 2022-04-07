import argparse
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from model import *
from diff_augs import DiffAugment
from augmentations import *
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import os
import configargparse


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument('--dataroot', type=str, default='dataset/obama', help='data folder to load from')
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
    parser.add_argument('--policy', type=str, default='test_stuff')
    parser.add_argument('--diff_augs', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='logs', help='saved logs')
    parser.add_argument('--exp_name', type=str, help='experiment name')


    args = parser.parse_args()
    print(args.policy, args.diff_augs)

    image_dataset = dataset.ImageFolder(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    # Create the generator
    netG = Generator(args).to(device)
    if (device.type == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(args).to(device)
    if (device.type == 'cuda') and (args.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))
    netD.apply(weights_init)

    # DCGAN loss
    criterion = nn.BCELoss()
    # Checkpoint noise
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    log_path = os.path.join(args.log_dir, args.exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    # Lists to keep track of progress

    schedulerD = StepLR(optimizerD, step_size=10000, gamma=0.6)
    schedulerG = StepLR(optimizerG, step_size=10000, gamma=0.6)

    print("Starting Training Loop...")
    for epoch in range(args.num_epochs):
        for i, data in enumerate(dataloader, 0):
            # update D
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            if args.diff_augs == True:
                output = netD(Augment(real_cpu, policy=args.policy))
                output = torch.squeeze(output)
            else:
                output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch           
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            # Generate fake image batch with G
            if args.diff_augs == True:
                fake = Augment(netG(noise), policy=args.policy)
                fake = torch.squeeze(fake)
            else:
                fake = netG(noise)

            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            if args.diff_augs == True:
                output = netD(Augment(fake, policy=args.policy)).view(-1)
                output = torch.squeeze(output)
            else:
                output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            schedulerD.step()
            schedulerG.step()


        if epoch % 100 == 0 or (epoch == args.num_epochs-1):
            writer.add_scalar("D_reals", D_x, epoch)
            writer.add_scalar("D_fake", D_G_z1, epoch)
            writer.add_scalar("Loss D", errD.item(), epoch)
            writer.add_scalar("Loss G", errG.item(), epoch) 
            writer.add_scalars("Overall Losses", {'D': errD.item(), 'G': errG.item()}, epoch)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, args.num_epochs, i, len(dataloader),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Check how the generator is doing by saving G's output on fixed_noise
        if (epoch % 10000 == 0) or (epoch == args.num_epochs-1):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            vutils.save_image(vutils.make_grid(fake[:8], normalize=True), os.path.join(log_path, 'img' + str(epoch) +'.png'))
            # save both discriminator and generator model   
            torch.save(netG.state_dict(), os.path.join(log_path, 'netG' + str(epoch) + '.pth'))
            torch.save(netD.state_dict(), os.path.join(log_path, 'netD' + str(epoch) + '.pth'))


    torch.save(netG.state_dict(), os.path.join(log_path, 'netG_recent.pth'))
    torch.save(netD.state_dict(), os.path.join(log_path, 'netD_recent.pth'))
    writer.flush()
    writer.close()