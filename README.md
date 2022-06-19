# Generative Deep Learning
### Teaching Machines to paint, write, compose and play

The official code repository for examples in the O'Reilly book 'Generative Deep Learning'

https://learning.oreilly.com/library/view/generative-deep-learning/9781492041931/

https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947/ref=sr_1_1

## Tensorflow 2.0

This branch uses Keras within Tensorflow 2.0.

## Structure

This repository is structured as follows:

The notebooks for each chapter are in the root of the repository, prefixed with the chapter number.

The `data` folder is where to download relevant data sources (chapter 3 onwards)
The `run` folder stores output from the generative models (chapter 3 onwards)
The `utils` folder stores useful functions that are sourced by the main notebooks

## Book Contents
Part 1: Introduction to Generative Deep Learning
* Chapter 1: Generative Modeling
* Chapter 2: Deep Learning
* Chapter 3: Variational Autoencoders
* Chapter 4: Generative Adversarial Networks

Part 2: Teaching Machines to Paint, Write, Compose and Play
* Chapter 5: Paint
* Chapter 6: Write
* Chapter 7: Compose
* Chapter 8: Play
* Chapter 9: The Future of Generative Modeling
* Chapter 10: Conclusion


## Getting started

To get started, first install the required libraries inside a virtual environment:

```
# install nvidia drivers if you haven't already:
#https://www.nvidia.com/Download/index.aspx

# make a tensorflow environment that works with
# 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz   2.50 GHz
# check your card:
nvidia-smi --query-gpu=gpu_name --format=csv|tail -n 1
# NVIDIA GeForce RTX 3050 Ti Laptop GPU
#

# install mamba for faster package management:
# sometimes you have to repeat a mamba command, its still faster than conda
conda install -n base conda-forge::mamba

mamba create -n generative tensorflow-gpu -c conda-forge
conda activate generative

pip install tf-explain
# uninstall tensorflow so it's linked to the version your cudakit needs later on
pip uninstall tensorflow

# install tool to query your nvidia toolkit version
mamba install cuda-nvcc -c nvidia
nvcc --version

# Assuming nvcc version is 11.7: will bring correct cudnn, and libcusolver.so.11:
# WARNING: 2.5.0 has a broken libcusolver
#   If you install cudatoolkit any other way for 2.5.0, libcusolver.so.10 will be installed when you need so.11, and you'll get errors
mamba install cudatoolkit=11.7 


### test:
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
# ignore NUMA node warnings, they're harmless, see: https://forums.developer.nvidia.com/t/numa-error-running-tensorflow-on-jetson-tx2/56119/2
# I think this happens if GPU is number '0'
# All the libs should load
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# ignore NUMA node warning

mamba install jupyter
mamba install --file requirements.txt -c conda-forge -c esri

# start your notebook server
LD_LIBRARY_PATH=~/miniconda3/envs/generative/lib jupyter notebook

```
 



