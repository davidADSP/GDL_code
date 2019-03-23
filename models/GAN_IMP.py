
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from keras.layers.merge import _Merge

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
from keras.initializers import RandomNormal

from utils.callbacks import CustomCallback, step_decay_schedule

import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class GAN():
    def __init__(self
        , input_dim
        , discriminator_conv_filters
        , discriminator_conv_kernel_size
        , discriminator_conv_strides
        , discriminator_conv_paddings
        , generator_conv_t_filters
        , generator_conv_t_kernel_size
        , generator_conv_t_strides
        , generator_conv_t_paddings
        , z_dim
        , learning_rate = 5e-5
        , use_batch_norm = True

        ):

        self.name = 'gan'

        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_conv_paddings = discriminator_conv_paddings
        self.generator_conv_t_filters = generator_conv_t_filters
        self.generator_conv_t_kernel_size = generator_conv_t_kernel_size
        self.generator_conv_t_strides = generator_conv_t_strides
        self.generator_conv_t_paddings = generator_conv_t_paddings
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate

        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_t_filters)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.discriminator, self.generator, self.model = self._build()

        

    def wasserstein(self, y_true, y_pred):
        return - K.mean(y_true * y_pred)

    def _build_discriminator(self):

        ### THE discriminator
        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')

        x = discriminator_input

        for i in range(self.n_layers_discriminator):
            conv_layer = Conv2D(
                filters = self.discriminator_conv_filters[i]
                , kernel_size = self.discriminator_conv_kernel_size[i]
                , strides = self.discriminator_conv_strides[i]
                , padding = self.discriminator_conv_paddings[i]
                , name = 'discriminator_conv_' + str(i)
                , use_bias = False
                , kernel_initializer=self.weight_init
                )

            x = conv_layer(x)

            x = LeakyReLU()(x)

            if i==1:
                x = ZeroPadding2D(padding=((0,1),(0,1)))(x)

            if self.use_batch_norm:
                x = BatchNormalization(momentum = 0.8)(x)

            
            # x = Dropout(rate = 0.25)(x)


        x = Flatten()(x)
        
        discriminator_output = Dense(1, use_bias=False)(x)

        discriminator = Model(discriminator_input, discriminator_output)

        return discriminator

    def _build_generator(self):

        ### THE generator

        generator_input = Input(shape=(self.z_dim,), name='generator_input')

        x = Dense(64 * 7 * 7)(generator_input)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = Reshape((7, 7, 64))(x)

        for i in range(self.n_layers_generator):
            conv_t_layer = Conv2DTranspose(
                filters = self.generator_conv_t_filters[i]
                , kernel_size = self.generator_conv_t_kernel_size[i]
                # , strides = self.generator_conv_t_strides[i]
                , padding = self.generator_conv_t_paddings[i]
                , name = 'generator_conv_t_' + str(i)
                , use_bias = False
                , kernel_initializer=self.weight_init
                )
            
            if i < self.n_layers_generator - 2:
                x = UpSampling2D()(x)
                x = conv_t_layer(x)
                x = Activation('relu')(x) #LeakyReLU()(x) #
                if self.use_batch_norm:
                    x = BatchNormalization(momentum = 0.8)(x)
                
                # x = Dropout(rate = 0.25)(x)
            else:
                x = conv_t_layer(x)
                x = Activation('tanh')(x)

            

        generator_output = x

        generator = Model(generator_input, generator_output)

        return generator


    def _build(self):
        
        discriminator = self._build_discriminator()
        generator = self._build_generator()

        ### COMPILE DISCRIMINATOR
    
        optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8) #RMSprop(lr=self.learning_rate) #Adam(lr=self.learning_rate)

        discriminator.compile(optimizer=optimizer, loss = self.wasserstein,  metrics = ['accuracy'])
        
        ### COMPILE THE FULL GAN

        discriminator.trainable = False

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = discriminator(generator(model_input))

        model = Model(model_input, model_output)

        model.compile(loss=self.wasserstein, optimizer=optimizer, metrics=['accuracy'])

        discriminator.trainable = True

        return discriminator, generator, model


    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.discriminator_conv_filters
                , self.discriminator_conv_kernel_size
                , self.discriminator_conv_strides
                , self.discriminator_conv_paddings
                , self.generator_conv_t_filters
                , self.generator_conv_t_kernel_size
                , self.generator_conv_t_strides
                , self.generator_conv_t_paddings
                , self.z_dim
                , self.use_batch_norm
                ], f)

        self.plot_model(folder)


    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 50, initial_epoch = 0, lr_decay = 1, discriminator_training_loops = 5, clip_threshold = 0.01):

        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)
        
        checkpoint_filepath=os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only = True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1)

        callbacks_list = [checkpoint1, checkpoint2, custom_callback, lr_sched]

        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(discriminator_training_loops):

                idx = np.random.randint(0, x_train.shape[0], batch_size)
                true_imgs = x_train[idx]
                
                noise = np.random.normal(0, 1, (batch_size, self.z_dim))

                gen_imgs = self.generator.predict(noise)

                # d_loss = self.discriminator.train_on_batch(np.concatenate([true_imgs, gen_imgs]), np.concatenate([valid, fake]))
                d_loss_real = self.discriminator.train_on_batch(true_imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
                    l.set_weights(weights)

            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            g_loss = self.model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, -d_loss[0], -g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % print_every_n_batches == 0:
                self.sample_images(epoch, run_folder)

    def sample_images(self, epoch, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray_r')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % epoch))
        plt.close()




    
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.discriminator, to_file=os.path.join(run_folder ,'viz/discriminator.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.generator, to_file=os.path.join(run_folder ,'viz/generator.png'), show_shapes = True, show_layer_names = True)



        


        

        


