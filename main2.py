import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import utils

from build_gan import Model

FLAGS = None

def main():
    BATCH_SIZE = FLAGS.batch_size
    IMG_SIZE = FLAGS.img_size
    # DATA_PATH = FLAGS.data_dir
    CUDA = FLAGS.cuda
    # OUT_PATH = 'output_glomerulus_256' # path to store output files

    # init GPU or CPU
    device = torch.device("cuda:0" if CUDA else "cpu")

    if FLAGS.train:
        print('Loading data...\n')

        # path to input data
        DATA_PATH = '~/pytorch_dcgan/train_glomerulus/'
        MNIST = True
        if MNIST:
            DATA_PATH = '~/pytorch_dcgan/train_mnist/mnist'

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
                                           # transforms.CenterCrop(X_DIM),
                                           # transforms.RandomHorizontalFlip(),
                                           # transforms.RandomVerticalFlip(),
                                           # transforms.RandomRotation(180),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                       ]))
        assert dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                 shuffle=True, num_workers=4, pin_memory=True)
        print('Creating model...\n')
        model = Model(FLAGS.model, device, dataloader, FLAGS.classes, FLAGS.channels, FLAGS.img_size, FLAGS.latent_dim)
        model.create_optim(FLAGS.lr)

        # Train
        model.train(FLAGS.epochs, FLAGS.log_interval, FLAGS.out_dir, True)

        model.save_to('')
    else:
        model = Model(FLAGS.model, device, None, FLAGS.classes, FLAGS.channels, FLAGS.img_size, FLAGS.latent_dim)
        model.load_from(FLAGS.out_dir)
        model.eval(mode=1, batch_size=FLAGS.batch_size)


if __name__ == '__main__':
    from utils import boolean_string
    parser = argparse.ArgumentParser(description='Hands-On GANs - Chapter 5')
    parser.add_argument('--model', type=str, default='cgan', help='one of `cgan` and `infogan`.')
    parser.add_argument('--cuda', type=boolean_string, default=True, help='enable CUDA.')
    parser.add_argument('--train', type=boolean_string, default=True, help='train mode or eval mode.')
    parser.add_argument('--data_dir', type=str, default='~/Data/mnist', help='Directory for dataset.')
    parser.add_argument('--out_dir', type=str, default='output_cgan_test', help='Directory for output.')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='size of batches')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='latent space dimension')
    parser.add_argument('--classes', type=int, default=10, help='number of classes')
    parser.add_argument('--img_size', type=int, default=256, help='size of images')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--log_interval', type=int, default=100, help='interval between logging and image sampling')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()

    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        if FLAGS.cuda:
            torch.cuda.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    cudnn.benchmark = True

    if FLAGS.train:
        utils.clear_folder(FLAGS.out_dir)
        # if FLAGS.model == 'infogan':
        #     utils.clear_folder(os.path.join(FLAGS.out_dir, 'c1'))
        #     utils.clear_folder(os.path.join(FLAGS.out_dir, 'c2'))
        #     utils.clear_folder(os.path.join(FLAGS.out_dir, 'c3'))

    log_file = os.path.join(FLAGS.out_dir, 'log.txt')
    print("Logging to {}\n".format(log_file))
    sys.stdout = utils.StdOut(log_file)

    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA version: {}\n".format(torch.version.cuda))

    print(" " * 9 + "Args" + " " * 9 + "|    " + "Type" + \
          "    |    " + "Value")
    print("-" * 50)
    for arg in vars(FLAGS):
        arg_str = str(arg)
        var_str = str(getattr(FLAGS, arg))
        type_str = str(type(getattr(FLAGS, arg)).__name__)
        print("  " + arg_str + " " * (20-len(arg_str)) + "|" + \
              "  " + type_str + " " * (10-len(type_str)) + "|" + \
              "  " + var_str)

    main()
