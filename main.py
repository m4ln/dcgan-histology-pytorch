# Script inspired by
# Hany, J. W. (2019).
# Hands-on Generative Adversarial Networks with PyTorch 1.0 :
# implement next-generation neural networks to build powerful gan models using python.

# IMPORTS
import os
import sys
from pathlib import Path

import numpy as np
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import utils
from dcgan import Generator, Discriminator, weights_init

def check_os():
    # check os for sds path
    if sys.platform == "linux":
        path1 = '/home/marlen/sds_hd/sd18a006/'
        path2 = '/home/mr38/sds_hd/sd18a006/'
        if Path(path1).exists():
            return path1
        elif Path(path2).exists():
            return path2
        else:
            print('error: sds path cannot be defined! Abort')
            return 1
    elif sys.platform == "win32":
        path = '//lsdf02.urz.uni-heidelberg.de/sd18A006/'
        if Path(path).exists():
            return path
        else:
            print('error: sds path cannot be defined! Abort')
            return 1
    else:
        print('error: sds path cannot be defined! Abort')
        return 1

if __name__ == "__main__":
    # if data path is sds
    # sds_path = check_os()

    ###################
    # HYPERPARAMETERS #
    ###################
    PHASE = 'train'
    # True for GPU training, False for CPU training
    CUDA = True
    # path to input data
    DATA_PATH = '~/pytorch_dcgan/train_glomerulus_01/'
    MNIST = False
    if MNIST:
        DATA_PATH = '~/pytorch_dcgan/train_mnist/mnist'
    OUT_PATH = 'output_glomerulus_01_200' # path to store output files
    # number of images in one batch, adjust this value according to your GPU memory
    BATCH_SIZE = 128
    # number if epochs for training (increase value for better results)
    EPOCH_NUM = 200
    # learning rate (increase value for better results)
    lr = 2e-4
    # number of channels, 1 for grayscale, 3 for rgb image
    IMAGE_CHANNEL = 3
    Z_DIM = 100
    IMG_SIZE = 64
    G_HIDDEN = 64
    X_DIM = 64
    D_HIDDEN = 64
    # labels for classification (1=real, 0=fake)
    REAL_LABEL = 1
    FAKE_LABEL = 0
    # Change to None to get different results at each run
    seed = 1

    # create log file and write outputs
    LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
    utils.clear_folder(OUT_PATH)
    print("Logging to {}\n".format(LOG_FILE))
    sys.stdout = utils.StdOut(LOG_FILE)
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))

    if seed is None:
        seed = np.random.randint(1, 10000)
    print("Random Seed: {}\n".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True      # May train faster but cost more memory

    # training parameters
    print("Learning rate: {}".format(lr))
    print("Batch size: {}".format(BATCH_SIZE))
    print("Epochs: {}\n".format(EPOCH_NUM))

    # load dataset
    if MNIST:
        dataset = dset.MNIST(root=DATA_PATH, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(IMG_SIZE),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))
                             ]))
    else:
        dataset = dset.ImageFolder(root=DATA_PATH,
                                   transform=transforms.Compose([
                                       transforms.Resize(IMG_SIZE),
                                       transforms.CenterCrop(X_DIM),
                                       # transforms.RandomHorizontalFlip(),
                                       # transforms.RandomVerticalFlip(),
                                       # transforms.RandomRotation(180),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4)

    # init GPU or CPU
    device = torch.device("cuda:0" if CUDA else "cpu")

    if(PHASE == 'train'):
        # Generator
        netG = Generator(IMAGE_CHANNEL, Z_DIM, G_HIDDEN).to(device)
        netG.apply(weights_init)
        print(netG)

        # Discriminator
        netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN).to(device)
        netD.apply(weights_init)
        print(netD)
    else:
        # Generator
        netG = Generator(IMAGE_CHANNEL, Z_DIM, G_HIDDEN)
        netG.load_state_dict(torch.load('model_ResNet152.pt'))
        netG.to(device)

        # Discriminator
        # netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN)
        netD = torchvision.models.resnet152(pretrained=True)
        netD.modName = 'ResNet152_loaded '
        # netD = torch.load('model_ResNet152.pt')
        netD.to(device)

    # loss function
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

    # optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(EPOCH_NUM):
        for i, data in enumerate(dataloader):
            x_real = data[0].to(device)
            real_label = torch.full((x_real.size(0),), REAL_LABEL, dtype=torch.float32, device=device)
            fake_label = torch.full((x_real.size(0),), FAKE_LABEL, dtype=torch.float32, device=device)

            # Update D with real data
            netD.zero_grad()
            y_real = netD(x_real)
            loss_D_real = criterion(y_real, real_label)
            loss_D_real.backward()

            # Update D with fake data
            z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device)
            x_fake = netG(z_noise)
            y_fake = netD(x_fake.detach())
            loss_D_fake = criterion(y_fake, fake_label)
            loss_D_fake.backward()
            optimizerD.step()

            # Update G with fake data
            netG.zero_grad()
            y_fake_r = netD(x_fake)
            loss_G = criterion(y_fake_r, real_label)
            loss_G.backward()
            optimizerG.step()

            if i % 100 == 0:
                print('Epoch {} [{}/{}] loss_D_real: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(
                    epoch, i, len(dataloader),
                    loss_D_real.mean().item(),
                    loss_D_fake.mean().item(),
                    loss_G.mean().item()
                ))
                vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_samples.png'), normalize=True)
                with torch.no_grad():
                    viz_sample = netG(viz_noise)
                    vutils.save_image(viz_sample, os.path.join(OUT_PATH, 'fake_samples_{}.png'.format(epoch)), normalize=True)
        torch.save(netG.state_dict(), os.path.join(OUT_PATH, 'netG_{}.pth'.format(epoch)))
        torch.save(netD.state_dict(), os.path.join(OUT_PATH, 'netD_{}.pth'.format(epoch)))
