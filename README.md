# Generative Deep Learning
### Objective
Reimplement current demos of "Deep Generative Learning" in PyTorch.

### [Temporary Draft] Notes
To make the codebase works in Windows, do the following:
1. ```conda install matplotlib```
2. ```conda install -c anaconda pillow==6.2.1```  
    - stepwise sanity check: 
    ```
    >> import matplotlib.pyplot as plt
    >> plt.plot([1,2])
    >> plt.savefig('test.jpg)
    ``` 
3. ```conda install graphviz``` and add graphviz bin to ```PATH```
4. ```conda install -c anaconda pyyaml```
5. ```conda install -c anaconda tensorflow-gpu==1.14.0```  
    - stepwise sanity check:
    ```
    >> import tensorflow as tf
    >> tf.test.is_gpu_available()
    ```
6. ```conda install -c conda-forge google-pasta```
7. ```conda install -c conda-forge keras==2.2.4```
8. install ```keras-contrib```:
    ```
    >> git clone https://www.github.com/keras-team/keras-contrib.git
    >> cd keras-contrib
    >> python setup.py install
    ``` 
    - stepwise sanity check:
    ```
    >> from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
    ```
9. install the rest in ```pip```: ```pip install -r requirement.txt```:  
    - stepwise sanity check: make sure you can run through ```03_01_autoencoder_train.ipynb``` (keras version)
10. ```conda install pytorch torchvision -c pytorch```:  
    - stepwise sanity check:
    ```
    >> import torch
    >> torch.cuda.is_available()
    ```
11. ```pip install torchsummary```

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


## [TBC] Getting started
Coming...



