import sys
from pathlib import Path

from datetime import datetime

import numpy as np

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from scipy.interpolate import interp1d

import utils
from dcgan import Generator, Discriminator, weights_init

if __name__ == "__main__":
    # HYPERPARAMETERS
    # path to project
    PROJECT_PATH = Path('D:/pytorch_dcgan/output/glomerulus/01')
    # path to input date
    IN_PATH = PROJECT_PATH.joinpath('train/netG_199.pth')
    # path to store output
    OUT_PATH = PROJECT_PATH.joinpath('test')
    # number of iterations to create images
    IMAGE_COUNT = 50
    # number of images in one batch, adjust this value according to your GPU memory
    BATCH_SIZE = 10
    # number of channels, 1 for grayscale, 3 for rgb image
    IMAGE_CHANNEL = 3
    Z_DIM = 100
    G_HIDDEN = 64
    # Change to None to get different results at each run
    SEED = None
    # 0: random; 1: interpolation; 2: semantic calculation
    VIZ_MODE = 0
    # True for GPU training, False for CPU training
    CUDA = True

    # create log file and write outputs
    # LOG_FILE = OUT_PATH.joinpath('log.txt')
    utils.clear_folder(OUT_PATH)
    # print("Logging to {}\n".format(LOG_FILE))
    # sys.stdout = utils.StdOut(LOG_FILE)
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))

    if SEED is None:
        SEED = np.random.randint(1, 10000)
    print("Random Seed: {}\n".format(SEED))
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if CUDA:
        torch.cuda.manual_seed(SEED)
    cudnn.benchmark = True      # May train faster but cost more memory

    # init GPU or CPU
    device = torch.device("cuda:0" if CUDA else "cpu")

    # Generator
    netG = Generator(IMAGE_CHANNEL, Z_DIM, G_HIDDEN)
    netG.load_state_dict(torch.load(IN_PATH))
    netG.to(device)

    for i in range(IMAGE_COUNT):
        if VIZ_MODE == 0:
            viz_tensor = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
        elif VIZ_MODE == 1:
            load_vector = np.loadtxt('vec_20190317-223131.txt')
            xp = [0, 1]
            yp = np.vstack([load_vector[2], load_vector[9]])   # choose two exemplar vectors
            xvals = np.linspace(0, 1, num=BATCH_SIZE)
            sample = interp1d(xp, yp, axis=0)
            viz_tensor = torch.tensor(sample(xvals).reshape(BATCH_SIZE, Z_DIM, 1, 1), dtype=torch.float32, device=device)
        elif VIZ_MODE == 2:
            load_vector = np.loadtxt('vec_20190317-223131.txt')
            z1 = (load_vector[0] + load_vector[6] + load_vector[8]) / 3.
            z2 = (load_vector[1] + load_vector[2] + load_vector[4]) / 3.
            z3 = (load_vector[3] + load_vector[4] + load_vector[6]) / 3.
            z_new = z1 - z2 + z3
            sample = np.zeros(shape=(BATCH_SIZE, Z_DIM))
            for i in range(BATCH_SIZE):
                sample[i] = z_new + 0.1 * np.random.normal(-1.0, 1.0, 100)
            viz_tensor = torch.tensor(sample.reshape(BATCH_SIZE, Z_DIM, 1, 1), dtype=torch.float32, device=device)

        with torch.no_grad():
            viz_sample = netG(viz_tensor)
            # viz_vector = utils.to_np(viz_tensor).reshape(BATCH_SIZE, Z_DIM)
            # cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            # np.savetxt(OUT_PATH.joinpath('vec_{}.txt'.format(i)), viz_vector)
            vutils.save_image(viz_sample, OUT_PATH.joinpath('img_{}.png'.format(i)), nrow=10, normalize=True)