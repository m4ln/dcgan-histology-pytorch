# DCGAN in PyTorch for Histopathological Image Generation
An implementation of a DCGAN (Deep Convolutional Generative Adversarial Network) for the generation of histopathological images, i.e classified glomerulus pathologies of the kidney.
The Code is derived from Chapter 4 & 5 from [Hany, J. & Walters, G. (2019). Hands-on Generative Adversarial Networks with PyTorch 1.0 : implement next-generation neural networks to build powerful gan models using python](https://github.com/PacktPublishing/Hands-On-Generative-Adversarial-Networks-with-PyTorch-1.x#hands-on-generative-adversarial-networks-with-pytorch-10).

## Prerequisites
- Linux OS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
    ```bash
    git clone https://github.com/m4ln/pytorch_dcgan.git
    cd pytorch_dcgan
    ```

- Install dependencies via [pip](https://pypi.org/project/pip/)
  ```
  pip install -r requirements.txt
  ```
  It might be necessary to install PyTorch manually from https://pytorch.org/get-started/locally/

### Training
- Change the hyperparameters inside `train.py` (Line 21-43)
- You can train the model on the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) dataset by not providing a valid dataset path
- To train on your own data change `IN_PATH` (Line 24) to your corresponding path, it should contain a subfolder of images for each class
- Train a model:
  ```bash
  python train.py
  ```
  
### Generating Fake Samples/Testing
- Change the hyperparameters inside `train.py` (Line 21-38)
- To generate new samples run:
  ```bash
  python test.py
  ```