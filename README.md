# DCGAN in PyTorch for Histopathological Image Generation
This repository implements a DCGAN (Deep Convolutional Generative Adversarial Network) for generating histopathological images, specifically glomerulus pathologies of the kidney represented by 12 classes. The code is adapted from Chapter 4 & 5 of [Hands-on Generative Adversarial Networks with PyTorch 1.0](https://github.com/PacktPublishing/Hands-On-Generative-Adversarial-Networks-with-PyTorch-1.x#hands-on-generative-adversarial-networks-with-pytorch-10) by Hany, J. & Walters, G. (2019).

## Prerequisites
- Linux OS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repository:
    ```bash
    git clone https://github.com/m4ln/pytorch_dcgan.git
    cd pytorch_dcgan
    ```

- Install dependencies via [pip](https://pypi.org/project/pip/)
  ```
  pip install -r requirements.txt
  ```
  Note: It might be necessary to install PyTorch manually from https://pytorch.org/get-started/locally/

### Training
- If no input arguments are provided, the model is trained on the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) dataset
  ```bash
  python train.py
  ```
- To train on your own data, provide the arguments via argparse (check inside `train.py`)
- The directory to your input data should contain a subfolder of images for each class
  
### Generating Fake Samples/Testing
- To generate new samples, run (by default using the MNIST trained model as in `train.py`):
  ```bash
  python test.py
  ```
- To test on your own data, provide the arguments via argparse (check inside `test.py`)

### Citation
If you use this project for your research, please cite our [paper](https://doi.org/10.1007/s40620-021-01221-9).
```
@article{weis2022assessment,
  title={Assessment of glomerular morphological patterns by deep learning algorithms},
  author={Weis, Cleo-Aron and Bindzus, Jan Niklas and Voigt, Jonas and Runz, Marlen and Hertjens, Svetlana and Gaida, Matthias M and Popovic, Zoran V and Porubsky, Stefan},
  journal={Journal of Nephrology},
  volume={35},
  number={2},
  pages={417--427},
  year={2022},
  doi = {10.11588/data/8LKEZF},
  publisher={Springer}
}
```
