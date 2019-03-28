from __future__ import print_function, division
import scipy

from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.merge import add
from models.layers.layers import ReflectionPadding2D
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K

from keras.utils import plot_model

import datetime
import matplotlib.pyplot as plt
import sys

import numpy as np
import os
import pickle as pkl
import random

from collections import deque


class CycleGAN():
    def __init__(self
        , input_dim
        , learning_rate
        , buffer_max_length
        , validation_weight
        , lambda_cycle
        , lambda_id
        ):

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.buffer_max_length = buffer_max_length
        self.validation_weight = validation_weight
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id

        # Input shape
        self.img_rows = input_dim[0]
        self.img_cols = input_dim[1]
        self.channels = input_dim[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        self.buffer_A = deque(maxlen = self.buffer_max_length)
        self.buffer_B = deque(maxlen = self.buffer_max_length)
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**3)
        self.disc_patch = (patch, patch, 1)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        optimizer = Adam(self.learning_rate, 0.5)


        self.compile_models()

        
    def compile_models(self):

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        
        self.d_A.compile(loss='mse',
            optimizer=Adam(self.learning_rate, 0.5),
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=Adam(self.learning_rate, 0.5),
            metrics=['accuracy'])


        # Build the generators
        self.g_AB = self._build_generator()
        self.g_BA = self._build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  self.validation_weight, self.validation_weight,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=Adam(0.0002, 0.5))

        self.d_A.trainable = True
        self.d_B.trainable = True
    

    def _build_generator(self):

        def c7s1_k(y, k, final):
            y = ReflectionPadding2D(padding =(3,3))(y)
            y = Conv2D(k, kernel_size=(7,7), strides=1, padding='valid', kernel_initializer = self.weight_init)(y)
            if final:
                y = Activation('tanh')(y)
            else:
                y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
                # y = ELU()(y)
                y = Activation('relu')(y)
            return y

        def d_k(y,k):
            y = Conv2D(k, kernel_size=(3,3), strides=2, padding='same', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
            # y = ELU()(y)
            y = Activation('relu')(y)
            return y

        def R_k(y, k):
            shortcut = y
            y = ReflectionPadding2D(padding =(1,1))(y)
            # down-sampling is performed with a stride of 2
            y = Conv2D(k, kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
            # y = ELU()(y)
            y = Activation('relu')(y)
            
            y = ReflectionPadding2D(padding =(1,1))(y)
            y = Conv2D(k, kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)

            # y = Activation('relu')(y)

            return add([shortcut, y])

        def u_k(y,k):
            # y = UpSampling2D()(y)
            # y = Conv2D(k, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer = self.weight_init)(y)
            y = Conv2DTranspose(k, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
            # y = ELU()(y)
            y = Activation('relu')(y)
    
            return y


        # Image input
        d0 = Input(shape=self.img_shape)

        y = d0

        y = c7s1_k(y, 64, False)
        y = d_k(y, 128)
        y = d_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = u_k(y, 128)
        y = u_k(y, 64)
        y = c7s1_k(y, 3, True)
        output_img = y

   
        return Model(d0, output_img)


    def build_discriminator(self):

        def C_k(y,k, stride = 2, norm=True):
            y = Conv2D(k, kernel_size=(4,4), strides=stride, padding='same', kernel_initializer = self.weight_init)(y)
            
            if norm:
                y = InstanceNormalization(axis = -1, center = False, scale = False)(y)

            y = LeakyReLU(0.2)(y)
           
            return y

        img = Input(shape=self.img_shape)

        y = C_k(img, 64, stride = 2, norm = False)
        y = C_k(y, 128, stride = 2)
        y = C_k(y, 256, stride = 2)
        y = C_k(y, 512, stride = 1)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same',kernel_initializer = self.weight_init)(y)

        return Model(img, validity)

    def train_discriminators(self, imgs_A, imgs_B, valid, fake):

        # Translate images to opposite domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        self.buffer_B.append(fake_B)
        self.buffer_A.append(fake_A)

        fake_A_rnd = random.sample(self.buffer_A, min(len(self.buffer_A), len(imgs_A)))
        fake_B_rnd = random.sample(self.buffer_B, min(len(self.buffer_B), len(imgs_B)))

        # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
        dA_loss_fake = self.d_A.train_on_batch(fake_A_rnd, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = self.d_B.train_on_batch(fake_B_rnd, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss_total = 0.5 * np.add(dA_loss, dB_loss)

        return (
            d_loss_total[0]
            , dA_loss[0], dA_loss_real[0], dA_loss_fake[0]
            , dB_loss[0], dB_loss_real[0], dB_loss_fake[0]
            , d_loss_total[1]
            , dA_loss[1], dA_loss_real[1], dA_loss_fake[1]
            , dB_loss[1], dB_loss_real[1], dB_loss_fake[1]
        )

    def train_generators(self, imgs_A, imgs_B, valid):

        # Train the generators
        return self.combined.train_on_batch([imgs_A, imgs_B],
                                                [valid, valid,
                                                imgs_A, imgs_B,
                                                imgs_A, imgs_B])


    def train(self, data_loader, run_folder, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(self.epoch, epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch()):

                d_loss = self.train_discriminators(imgs_A, imgs_B, valid, fake)
                g_loss = self.train_generators(imgs_A, imgs_B, valid)

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                    % ( self.epoch, epochs,
                        batch_i, data_loader.n_batches,
                        d_loss[0], 100*d_loss[7],
                        g_loss[0],
                        np.sum(g_loss[1:3]),
                        np.sum(g_loss[3:5]),
                        np.sum(g_loss[5:7]),
                        elapsed_time))

                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(data_loader, batch_i, run_folder)
                    self.combined.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (self.epoch)))
                    self.save_model(run_folder)

                
            self.epoch += 1

    def sample_images(self, data_loader, batch_i, run_folder):
        
        r, c = 2, 4

        for p in range(2):

            if p == 1:
                imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
                imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)
            else:
                imgs_A = data_loader.load_img('data/%s/testA/test.jpg' % data_loader.dataset_name)
                imgs_B = data_loader.load_img('data/%s/testB/test.jpg' % data_loader.dataset_name)

            # Translate images to the other domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)
            # Translate back to original domain
            reconstr_A = self.g_BA.predict(fake_B)
            reconstr_B = self.g_AB.predict(fake_A)

            # ID the images
            id_A = self.g_BA.predict(imgs_A)
            id_B = self.g_AB.predict(imgs_B)

            gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, id_A, imgs_B, fake_A, reconstr_B, id_B])

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            gen_imgs = np.clip(gen_imgs, 0, 1)

            titles = ['Original', 'Translated', 'Reconstructed', 'ID']
            fig, axs = plt.subplots(r, c, figsize=(25,12.5))
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[j])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(run_folder ,"images/%d_%d_%d.png" % (p, self.epoch, batch_i)))
            plt.close()


    def plot_model(self, run_folder):
        plot_model(self.combined, to_file=os.path.join(run_folder ,'viz/combined.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.d_A, to_file=os.path.join(run_folder ,'viz/d_A.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.d_B, to_file=os.path.join(run_folder ,'viz/d_B.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.g_BA, to_file=os.path.join(run_folder ,'viz/g_BA.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.g_AB, to_file=os.path.join(run_folder ,'viz/g_AB.png'), show_shapes = True, show_layer_names = True)


    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim
                ,  self.learning_rate
                ,  self.buffer_max_length
                ,  self.validation_weight
                ,  self.lambda_cycle
                ,  self.lambda_id
                ], f)

        self.plot_model(folder)


    def save_model(self, run_folder):


        self.combined.save(os.path.join(run_folder, 'model.h5')  )
        self.d_A.save(os.path.join(run_folder, 'd_A.h5') )
        self.d_B.save(os.path.join(run_folder, 'd_B.h5') )
        self.g_BA.save(os.path.join(run_folder, 'g_BA.h5')  )
        self.g_AB.save(os.path.join(run_folder, 'g_AB.h5') )

        pkl.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))

    def load_weights(self, filepath):
        self.combined.load_weights(filepath)

if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=200, batch_size=1, sample_interval=100)
