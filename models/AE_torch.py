"""
REFERENCE:
1. [pytorch] autoencoder build: https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c
2. [keras] UpSampling2D, Conv2DTranspose difference with simple code: https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
3. [github] torch-summary repo: https://github.com/sksq96/pytorch-summary
    - can't handle single module: https://github.com/sksq96/pytorch-summary/issues/9
    - [medium] tutorial: https://medium.com/@umerfarooq_26378/model-summary-in-pytorch-b5a1e4b64d25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


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
        linear_input = self._get_conv_output(*input_shape)
        # bottleneck
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features = linear_input, out_features = z_dim)

    def _get_conv_output(self, c, h, w):
        # don't do autograd
        t = torch.rand(1, c, h, w, requires_grad = False)
        out = self._pseudo_forward(t)
        return out.view(out.size(0), -1).shape[1]

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
        out = self.flatten(out)
        out = self.linear(out)
        return out
        

class Decoder(nn.Module):
    def __init__(
        self, z_dim, 
        decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
        use_batch_norm, use_dropout
        ):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        self.n_layers_encoder = len(decoder_conv_t_filters)

    def forward(self, x):
        pass


class Autoencoder(nn.Module):
    def __init__(
        self, input_dim, 
        encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
        decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
        z_dim, use_batch_norm = False, use_dropout = False
        ):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(
            input_dim, 
            encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
            use_batch_norm, use_dropout
        )
        self.decoder = Decoder(
            z_dim, 
            decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides,
            use_batch_norm, use_dropout
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x