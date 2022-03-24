import argparse
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.optim as optim
import matplotlib
from model import *
from diff_augs import DiffAugment
from augmentations import hue, saturation, contrast
from torch.optim.lr_scheduler import StepLR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='dataset/obama', help='data folder to load from')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3, help='number of channels')
    parser.add_argument('--nz', type=int, default=100, help='dimension of latent')
    parser.add_argument('--ngf', default=64, help='size of feature maps in generator')
    parser.add_argument('--ndf', default=64, help='size of feature maps in discriminator')
    parser.add_argument('--num_epochs', default=1000, help='number of training epochs')
    parser.add_argument('--lr', default=0.0002, help='learning rate for optimizers')
    parser.add_argument('--beta1', default=0.5, help='beta1 hyperparam for Adam optimizers')
    parser.add_argument('--ngpu', default=1, help='number of GPUs available, 0 for CPU')
    parser.add_argument('--policy', type=str, default='color,translation,cutout')
    parser.add_argument('--diff_augs', type=bool, default=False)

    args = parser.parse_args()

    image_dataset = dataset.ImageFolder(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    print(device)

    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()

        # Create the generator
    netG = Generator(args).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator(args).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    schedulerD = StepLR(optimizerD, step_size=1000, gamma=0.7)
    schedulerG = StepLR(optimizerG, step_size=1000, gamma=0.7)

    print("Starting Training Loop...")
    for epoch in range(args.num_epochs):
        for i, data in enumerate(dataloader, 0):
            schedulerD.step()
            schedulerG.step()

            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            hue(real_cpu)
            saturation(real_cpu)
            contrast(real_cpu)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            if args.diff_augs == True:
                output = netD(DiffAugment(real_cpu, policy=args.policy))
                output = torch.squeeze(output)
            else:
                output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch           
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            # Generate fake image batch with G
            if args.diff_augs == True:
                fake = DiffAugment(netG(noise), policy=args.policy)
                fake = torch.squeeze(fake)
            else:
                fake = netG(noise)

            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
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
                output = netD(DiffAugment(fake, policy=args.policy)).view(-1)
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

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, args.num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()