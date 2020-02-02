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


class Encoder(nn.Module):
    def __init__(
        self, input_dim, 
        encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
        use_batch_norm, use_dropout
        ):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        self.n_layers_encoder = len(encoder_conv_filters)

    def forward(self, x):
        for i in range(self.n_layers_encoder):
            in_channels = self.input_dim if i == 0 else self.encoder_conv_filters[i-1]
            x = nn.Conv2d(
                in_channels = in_channels, out_channels = self.encoder_conv_filters[i], 
                kernel_size = self.encoder_conv_kernel_size[i], stride = self.encoder_conv_filters[i], 
                padding = self.encoder_conv_filters[i] // 2
                )(x)
            x = F.leaky_relu(x)
            if self.use_batch_norm:
                x = nn.BatchNorm2d(num_features = self.encoder_conv_filters[i])(x)
            if self.use_dropout:
                x = F.dropout(x, p = 0.25)
        return nn.Flatten()(x)


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
        x=  self.decoder(x)
        return x