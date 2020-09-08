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
    sds_path = check_os()

    # HYPERPARAMETERS
    PHASE = 'train'
    CUDA = True     # True for GPU training, False for CPU training
    # DATA_PATH = sds_path + '/DataBaseGlomerulusProjekt/KlassifikationDataBasePython'    # path to input data
    # DATA_PATH = sds_path + '/Marlen/train2'    # path to input data
    DATA_PATH = '~/pytorch_dcgan/train_glomerulus/'    # path to input data
    # DATA_PATH = '~\Data\mnist'
    OUT_PATH = 'output_glomerulus_256' # path to store output files
    LOG_FILE = os.path.join(OUT_PATH, 'log.txt')    # log file to record loss values
    BATCH_SIZE = 64        # number of images in one batch, adjust this value according to your GPU memory
    IMAGE_CHANNEL = 3   # 1 for grayscale, 3 for rgb image
    Z_DIM = 100
    dim = 256
    G_HIDDEN = 64
    X_DIM = dim
    D_HIDDEN = 64
    EPOCH_NUM = 100  # number if epochs for training
    REAL_LABEL = 1
    FAKE_LABEL = 0
    lr = 2e-4   # learning rate
    seed = 1    # Change to None to get different results at each run

    utils.clear_folder(OUT_PATH)
    print("Logging to {}\n".format(LOG_FILE))
    sys.stdout = utils.StdOut(LOG_FILE)
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))

    if seed is None:
        seed = np.random.randint(1, 10000)
    print("Random Seed: ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True      # May train faster but cost more memory

    print("Learning rate: ", lr)
    print("Batch size: ", BATCH_SIZE)

    # trfm_mean = [0.485, 0.456, 0.406]
    # trfm_std = [0.229, 0.224, 0.225]
    trfm_mean = [0.5, 0.5, 0.5]
    trfm_std = [0.5, 0.5, 0.5]
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.Resize(X_DIM),
    #         transforms.CenterCrop(X_DIM),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomVerticalFlip(),
    #         transforms.RandomRotation(180),
    #         transforms.ToTensor(),
    #         transforms.Normalize(trfm_std, trfm_std)
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(X_DIM),
    #         transforms.CenterCrop(X_DIM),
    #         transforms.ToTensor(),
    #         transforms.Normalize(trfm_mean, trfm_std)
    #     ]),
    # }
    #
    # image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(DATA_PATH, x),
    #                                           data_transforms[x])
    #                   for x in ['train', 'val']}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
    #                                              shuffle=True, num_workers=4)
    #               for x in ['train', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # dataloader = dataloaders[PHASE]

    dataset = dset.ImageFolder(root=DATA_PATH,
                               transform=transforms.Compose([
                                   transforms.Resize(X_DIM),
                                   transforms.CenterCrop(X_DIM),
                                   # transforms.RandomHorizontalFlip(),
                                   # transforms.RandomVerticalFlip(),
                                   # transforms.RandomRotation(180),
                                   transforms.ToTensor(),
                                   transforms.Normalize(trfm_mean, trfm_std),
                               ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4)

    # MNIST dataset
    # dataset = dset.MNIST(root=DATA_PATH, download=True,
    #                      transform=transforms.Compose([
    #                      transforms.Resize(X_DIM),
    #                      transforms.ToTensor(),
    #                      transforms.Normalize((0.5,), (0.5,))
    #                      ]))
    # assert dataset
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
    #                                          shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if CUDA else "cpu")

    # if(PHASE == 'train'):
    netG = Generator(IMAGE_CHANNEL, Z_DIM, G_HIDDEN).to(device)
    netG.apply(weights_init)
    print(netG)
    # else:
    #     netG = Generator(IMAGE_CHANNEL, Z_DIM, G_HIDDEN)
    #     netG.load_state_dict(torch.load('model_ResNet152.pt'))
    #     netG.to(device)

    if (PHASE == 'train'):
        netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN).to(device)
        netD.apply(weights_init)
        print(netD)
    else:
        # netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN)
        netD = torchvision.models.resnet152(pretrained=True)
        netD.modName = 'ResNet152_loaded '
        # netD = torch.load('model_ResNet152.pt')
        netD.to(device)

    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(EPOCH_NUM):
        # i=-1
        # for inputs, labels in dataloader:
        #         i=i+1
        for i, data in enumerate(dataloader):
            # x_real = inputs.to(device)
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
