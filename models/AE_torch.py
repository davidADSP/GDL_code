"""
DIFFERENCE WITH KERAS VERSION:
1. no callbacks yet
2. no plot of model architecture

REFERENCE:
1. [pytorch] autoencoder build: https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c
2. [keras] UpSampling2D, Conv2DTranspose difference with simple code: https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
3. [github] torch-summary repo: https://github.com/sksq96/pytorch-summary
    - can't handle single module: https://github.com/sksq96/pytorch-summary/issues/9
    - [medium] tutorial: https://medium.com/@umerfarooq_26378/model-summary-in-pytorch-b5a1e4b64d25
4. [github] visualization on convolution / convolution transpose operation: https://github.com/vdumoulin/conv_arithmetic
"""
import os
import sys

import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary


class TransConvBlock(nn.Module):
    def __init__(
            self, channel_in, channel_out, 
            kernel_size, stride, padding, 
            use_batch_norm, use_dropout
        ):
        super(TransConvBlock, self).__init__()
        pad_n = kernel_size // 2
        # output_padding control additional padding
        output_pad_n = 0 if stride == 1 else 1
        self.trans_conv = nn.ConvTranspose2d(in_channels = channel_in, out_channels = channel_out, 
                                             kernel_size = kernel_size, stride = stride, 
                                             padding = pad_n, output_padding = output_pad_n)
        self.leaky_relu = nn.LeakyReLU()
        # follow momentum default from keras (i.e. 0.99)
        self.bn = None
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(num_features = channel_out, momentum = 0.99)
        self.dropout = None
        if use_dropout:
            self.dropout = nn.Dropout(p = 0.25)

    def forward(self, x):
        out = self.trans_conv(x)
        out = self.leaky_relu(out)
        if self.bn is not None:
            out = self.bn(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ConvBlock(nn.Module):
    def __init__(
            self, channel_in, channel_out, 
            kernel_size, stride, padding, 
            use_batch_norm, use_dropout
        ):
        super(ConvBlock, self).__init__()
        # special case for kernel_size = 3 to have same padding
        pad_n = kernel_size // 2
        self.conv = nn.Conv2d(in_channels = channel_in, out_channels = channel_out, 
                              kernel_size = kernel_size, stride = stride, padding = pad_n)
        self.leaky_relu = nn.LeakyReLU()
        # follow momentum default from keras (i.e. 0.99)
        self.bn = None
        if use_batch_norm:
            self.bn = nn.BatchNorm2d(num_features = channel_out, momentum = 0.99)
        self.dropout = None
        if use_dropout:
            self.dropout = nn.Dropout(p = 0.25)

    def forward(self, x):
        out = self.conv(x)
        out = self.leaky_relu(out)
        if self.bn is not None:
            out = self.bn(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Encoder(nn.Module):
    def __init__(
            self, input_shape, 
            conv_filters, conv_kernel_size, conv_strides,
            z_dim, use_batch_norm, use_dropout
        ):
        """
        input_shape = (C, H, W) of input image
        z_dim = dimension of latent space
        """
        n = len(conv_kernel_size)
        assert len(conv_filters) == n, '[ERROR] encoder_conv_filters and encoder_conv_kernel_size must have same length'
        assert len(conv_strides) == n, '[ERROR] encoder_conv_kernel_size and encoder_conv_strides must have same length'
        super(Encoder, self).__init__()
        blk_ls = []
        for i, (channel_out, kernel_size, stride) in enumerate(zip(conv_filters, conv_kernel_size, conv_strides)):
            channel_in = input_shape[0] if i == 0 else conv_filters[i - 1]
            blk = ConvBlock(channel_in, channel_out, kernel_size, 
                            stride, kernel_size // 2, 
                            use_batch_norm, use_dropout)
            blk_ls.append(blk)
        self.blks = nn.ModuleList(blk_ls)
        # torch.Size object
        self.shape_before_flattening = self._get_conv_output(*input_shape)
        # bottleneck
        self.linear = nn.Linear(in_features = self.shape_before_flattening.numel(), 
                                out_features = z_dim)

    def _get_conv_output(self, c, h, w):
        """
        excluding batch size
        """
        # don't do autograd
        t = torch.rand(1, c, h, w, requires_grad = False)
        out = self._pseudo_forward(t)
        return out.shape[1:]

    def _pseudo_forward(self, x):
        """
        call a pseudo forward pass to calculate input size of nn.Linear
        """
        out = x
        for blk in self.blks:
            out = blk(out)
        return  out    

    def forward(self, x):
        out = x
        for blk in self.blks:
            out = blk(out)
        # reshape
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        

class Decoder(nn.Module):
    """
    input:
        reshape_shape -- torch.Size, shape to be reshaped after linear layer
    """    
    def __init__(
            self, reshape_shape, 
            conv_filters, conv_kernel_size, conv_strides,
            z_dim, use_batch_norm, use_dropout
        ):
        n = len(conv_kernel_size)
        assert len(conv_filters) == n, '[ERROR] encoder_conv_filters and encoder_conv_kernel_size must have same length'
        assert len(conv_strides) == n, '[ERROR] encoder_conv_kernel_size and encoder_conv_strides must have same length'
        assert isinstance(reshape_shape, torch.Size), '[ERROR] reshape_shape must be torch.Size'
        super(Decoder, self).__init__()
        self.reshape_shape = reshape_shape
        self.linear = nn.Linear(in_features = z_dim, out_features = reshape_shape.numel())
        blk_ls = []
        for i, (channel_out, kernel_size, stride) in enumerate(zip(conv_filters, conv_kernel_size, conv_strides)):
            channel_in = reshape_shape[0] if i == 0 else conv_filters[i - 1]
            blk = TransConvBlock(channel_in, channel_out, kernel_size, 
                                 stride, kernel_size // 2, 
                                 use_batch_norm, use_dropout)
            blk_ls.append(blk)
        self.blks = nn.ModuleList(blk_ls)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.size(0), *tuple(self.reshape_shape))
        for blk in self.blks:
            out = blk(out)
        return out


class Autoencoder(nn.Module):
    """
    input:
        input_shape -- tuple, (C, H, W)
        z_dim -- int, dimension of latent space
    """
    def __init__(
        self, input_shape, 
        encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
        decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
        z_dim, use_batch_norm = False, use_dropout = False
        ):
        super(Autoencoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape = input_shape
        self.latent_shape = (z_dim, )
        self.encoder = Encoder(
                            input_shape, 
                            encoder_conv_filters, encoder_conv_kernel_size, 
                            encoder_conv_strides, z_dim, 
                            use_batch_norm, use_dropout
                        )
        self.decoder = Decoder(
                            self.encoder.shape_before_flattening,
                            decoder_conv_t_filters, decoder_conv_t_kernel_size, 
                            decoder_conv_t_strides, z_dim, 
                            use_batch_norm, use_dropout
                        )
        self.to(self.device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encoder_summary(self):
        summary(self.encoder, input_size = self.input_shape)

    def decoder_summary(self):
        summary(self.decoder, input_size = self.latent_shape)

    def autoencoder_summary(self):
        summary(self, input_size = self.input_shape)


def compile(model, learning_rate):
    optimiser = optim.Adam(model.parameters(), lr = learning_rate)
    return optimiser

def save(model, folder):
    pass

def load_weights(model, filepath):
    # Load state dicts
    assert os.path.isfile(filepath), f'[ERROR] weights not exist: {filepath}' 
    model.load_state_dict(torch.load(filepath))
    return model


def train(model, train_ds, opt, batch_size, epochs, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1):
    z_dim = model.latent_shape[0]
    device = model.device
    # similar to step_decay_schedule
    scheduler = StepLR(opt, step_size = 1, gamma = lr_decay)
    loss_f = nn.MSELoss()

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle = True) 
    for epoch in range(epochs):
        loop = tqdm(train_loader)
        for batch, data in enumerate(loop):
            model.train()
            img, _ = data
            img = img.cuda()
            out = model(img)
            loss = loss_f(out, img)
            # progress bar
            loop.set_description(f'Epoch {epoch + 1}/{epochs}')
            loop.set_postfix(loss = loss.item())
            # backprop
            opt.zero_grad()
            loss.backward()
            opt.step()
            # similar to CustomCallback
            if batch % print_every_n_batches != 0:
                continue
            with torch.no_grad():
                model.eval()
                z_new = torch.randn(1, z_dim).to(device)
                reconst = model.decoder(z_new).cpu().numpy().squeeze()
                filepath = os.path.join(run_folder, 'images', f'img_{epoch:03}_{batch}.jpg')
                if len(reconst.shape) == 2:
                    plt.imsave(filepath, reconst, cmap = 'gray_r')
                else:
                    plt.imsave(filepath, reconst)
        if epoch >= initial_epoch: scheduler.step()
        # similar to ModelCheckpoint
        save_path = os.path.join(run_folder, 'weights', 'weight.pth')
        torch.save(model.state_dict(), save_path)
        print(f'\nEpoch {epoch + 1:05}: saving model to {save_path}')