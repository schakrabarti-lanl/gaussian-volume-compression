# 3D Gaussian Splatting for Volume Reconstruction
This repository contains an implementation of training a mixed Gaussian model from volumetric data. The codebase borrows heavily from the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering", which can be found [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@gitlab.newmexicoconsortium.org:ldyken/gaussian-volume.git --recursive
```
or
```shell
# HTTPS
git clone https://gitlab.newmexicoconsortium.org/ldyken/gaussian-volume.git --recursive
```

## Overview

The codebase consists of a PyTorch-based optimizer to produce a 3D Gaussian model from an input volumetric dataset (either unstructured .vtu or structured .vtk).

## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions
- CUDA SDK 11 for PyTorch extensions
- C++ Compiler and CUDA SDK must be compatible

### Setup

#### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate gaussian_splatting
```

Tip: Downloading packages and creating a new environment with Conda can require a significant amount of disk space. By default, Conda will use the main system hard drive. You can avoid this by specifying a different package download location and an environment on a different drive:

```shell
conda config --add pkgs_dirs <Drive>/<pkg_path>
conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
conda activate <Drive>/<env_path>/gaussian_splatting
```

#### Modifications

If you can afford the disk space, we recommend using our environment files for setting up a training environment identical to ours. If you want to make modifications, please note that major version changes might affect the results of our method. Make sure to create an environment where PyTorch and its CUDA runtime version match and the installed CUDA SDK has no major version difference with PyTorch's CUDA version.

### Running

To run the optimizer, simply use

```shell
python train.py -s <path to unstructured or structured volume dataset>
```

## Fixes
- *It breaks when trying to install diff-gaussian-rasterization or submodules/simple-knn during the creation from environment.yml. How do I proceed?* Just try again essentially, like so;
```
conda activate gaussian_splatting
cd <dir_to_repo>/gaussian-splatting
pip install diff-gaussian-rasterization
pip install submodules\simple-knn
```