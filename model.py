import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.ngpu = args.ngpu

        layers = []
        layers.append(nn.ConvTranspose2d(args.nz, args.ngf * 16, 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(args.ngf * 16))
        layers.append(nn.ReLU(True))
        # state size. (ngf*8) x 4 x 4
        layers.append(nn.ConvTranspose2d(args.ngf*16, args.ngf * 8, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(args.ngf * 8))
        layers.append(nn.ReLU(True))
        # state size. (ngf*4) x 8 x 8
        layers.append(nn.ConvTranspose2d(args.ngf*8, args.ngf * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(args.ngf * 4))
        layers.append(nn.ReLU(True))
        # state size. (ngf*2) x 16 x 16
        layers.append(nn.ConvTranspose2d( args.ngf * 4, args.ngf *2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(args.ngf * 2))
        layers.append(nn.ReLU(True))
        # state size. (ngf) x 32 x 32
        layers.append(nn.ConvTranspose2d( args.ngf * 2, args.ngf, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(args.ngf))
        layers.append(nn.ReLU(True))
        # state size. (ngf) x 64 x 64
        layers.append(nn.ConvTranspose2d( args.ngf, args.nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())
        # state size. (nc) x 128 x 128

        # self.main = nn.Sequential(*layers)


        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.ngpu = args.ngpu

        layers = []
        # input is (nc) x 128 x 128
        layers.append(nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(args.ndf))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf) x 64 x 64
        layers.append(nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(args.ndf * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf) x 32 x 32
        layers.append(nn.Conv2d(args.ndf*2, args.ndf * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(args.ndf * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf) x 16 x 16
        layers.append(nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(args.ndf * 8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf) x 8 x 8
        layers.append(nn.Conv2d(args.ndf * 8, args.ndf * 16, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(args.ndf * 16))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf*16) x 4 x 4
        layers.append(nn.Conv2d(args.ndf * 16, 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())

        # self.main = nn.Sequential(*layers)

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)