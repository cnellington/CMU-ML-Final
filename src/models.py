'''
    models.py

    Daniel Jeong
    danielje@cs.cmu.edu

    Provides implementation of AutoEncoder, VAE, Beta-VAE for Facial Expression Recognition (FER).

'''

import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import subprocess
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, Conv2DTranspose, Reshape
from tensorflow.keras.backend import random_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.regularizers import L1
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow_probability as tfp

# AutoEncoder
class AE:
    def __init__(self):
        super(AE, self).__init__()
    
        # Input layer
        self.input_img = Input(shape=(48,48,1), name='input_img')
        
        # Encoder
        self.encoder = Sequential([
            Conv2D(16, (3,3), padding='same', activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(32, (3,3), padding='same', activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), padding='same', activation='relu'),
            MaxPooling2D((2,2))
        ], name='encoderAE')
        
        # Decoder
        self.decoder = Sequential([
            Conv2DTranspose(16, (3,3), strides=2, padding='same',
                            activation='relu'),
            Conv2DTranspose(32, (3,3), strides=2, padding='same',
                            activation='relu'),
            Conv2DTranspose(64, (3,3), strides=2, padding='same',
                            activation='relu'),
            Conv2D(1, (3,3), padding='same', activation='sigmoid')
        ], name='decoderAE')
        
        # Define full model
        self.encodings = self.encoder(self.input_img)
        self.decodings = self.decoder(self.encodings)
        self.model = Model(inputs=self.input_img, outputs=self.decodings)
        
        # Compile model
        self.model.compile(optimizer=Adam(), loss='mse')
        
    def fit(self, inputs, epochs, batch_size, val_data=None, workers=1, callbacks=None):
        history = self.model.fit(inputs, inputs, batch_size=batch_size, epochs=epochs,
                                 validation_data=val_data, workers=workers,
                                 callbacks=callbacks)
        
        return history
        
    def get_encoder(self):
        encoder = Model(inputs=self.model.get_layer('encoderAE').input,
                        outputs=self.model.get_layer('encoderAE').output)
        
        return encoder
    
    def get_decoder(self):
        decoder = Model(inputs=self.model.get_layer('decoderAE').input,
                        outputs=self.model.get_layer('decoderAE').output)
        
        return decoder
        
    def summary(self):
        self.model.summary()
        
    def encoder_summary(self):
        encoder = self.get_encoder()
        encoder.compile()
        encoder.summary()
        
    def decoder_summary(self):
        decoder = self.get_decoder()
        decoder.compile()
        decoder.summary()
        
    def encode(self, inputs):
        encoder = self.get_encoder()
        encodings = encoder.predict(inputs)
        
        return encodings
    
    def decode(self, inputs):
        decoder = self.get_decoder()
        decodings = decoder.predict(inputs)
        
        return decodings

# Sampling layer with reparameterization trick
class Sampling(Layer):
    def __init__(self, name=None):
        super(Sampling, self).__init__(name=name)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dims = tf.shape(z_mean)[1]

        # Reparameterization Trick
        epsilon = random_normal(shape=(batch_size, latent_dims))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Scaled MSE
# Note: 900 = ~(28709 / 32)
def scaled_mse(y_true, y_pred, w=900):
    return w * mean_squared_error(y_true, y_pred)

# Variational AutoEncoder (VAE)
# Note: Providing a beta term makes it a beta-VAE
class VAE:
    def __init__(self, latent_dims, beta=1):
        super(VAE, self).__init__()
        self.latent_dims = latent_dims

        # Encoder
        encoder_input = Input(shape=(48,48,1), name='encoder_input')
        e = Conv2D(16, (3,3), padding='same', activation='relu')(encoder_input)
        e = MaxPooling2D((2,2))(e)
        e = Conv2D(32, (3,3), padding='same', activation='relu')(e)
        e = MaxPooling2D((2,2))(e)
        e = Conv2D(64, (3,3), padding='same', activation='relu')(e)
        e = MaxPooling2D((2,2))(e)
        e = Flatten()(e)

        z_mean = Dense(self.latent_dims, name='z_mean')(e)
        z_log_var = Dense(self.latent_dims, name='z_log_var')(e)
        z = Sampling(name='Sampling')((z_mean, z_log_var))

        self.encoder = Model(inputs=encoder_input, outputs=z, name='encoderVAE')

        # Decoder
        decoder_input = Input(shape=(latent_dims,), name='decoder_input')
        d = Dense(6*6*64, activation='relu')(decoder_input)
        d = Reshape(target_shape=(6,6,64))(d)
        d = Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu')(d)
        d = Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu')(d)
        d = Conv2DTranspose(16, (3,3), strides=2, padding='same', activation='relu')(d)
        d = Conv2DTranspose(1, (3,3), padding='same', activation='relu')(d)

        self.decoder = Model(inputs=decoder_input, outputs=d, name='decoderVAE')
        
        # Define full model
        output_img = self.decoder(z)
        self.model = Model(inputs=encoder_input, outputs=output_img, name='VAE')
        
        # Add KL loss
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1) * beta
        self.model.add_loss(kl_loss)

        # Compile model
        self.model.compile(optimizer=Adam(), loss=scaled_mse, metrics=[scaled_mse])
        
    def fit(self, inputs, epochs, batch_size, val_data=None, workers=1, callbacks=None):
        history = self.model.fit(inputs, inputs, batch_size=batch_size, epochs=epochs,
                                 validation_data=val_data, workers=workers,
                                 callbacks=callbacks)
        
        return history
        
    def get_encoder(self):
        encoder = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('Sampling').output)
        
        return encoder
    
    def get_decoder(self):
        decoder = Model(inputs=self.model.get_layer('decoderVAE').input,
                        outputs=self.model.get_layer('decoderVAE').output)
        
        return decoder
        
    def summary(self):
        self.model.summary()
        
    def encode(self, inputs):
        encoder = self.get_encoder()
        encodings = encoder.predict(inputs)
        
        return encodings
    
    def decode(self, inputs):
        decoder = self.get_decoder()
        decodings = decoder.predict(inputs)
        
        return decodings

# Variational AutoEncoder (VAE) with Emotion Decoder
# Note: Providing a beta term makes it a beta-VAE
# Note: Linear option changes emotion decoder to be linear for interpretability
class EmotionVAE:
    def __init__(self, latent_dims, beta=1, linear=False, sparse=False):
        super(EmotionVAE, self).__init__()
        self.latent_dims = latent_dims

        # Encoder
        encoder_input = Input(shape=(48,48,1), name='encoder_input')
        e = Conv2D(16, (3,3), padding='same', activation='relu')(encoder_input)
        e = MaxPooling2D((2,2))(e)
        e = Conv2D(32, (3,3), padding='same', activation='relu')(e)
        e = MaxPooling2D((2,2))(e)
        e = Conv2D(64, (3,3), padding='same', activation='relu')(e)
        e = MaxPooling2D((2,2))(e)
        e = Flatten()(e)

        z_mean = Dense(self.latent_dims, name='z_mean')(e)
        z_log_var = Dense(self.latent_dims, name='z_log_var')(e)
        z = Sampling(name='Sampling')((z_mean, z_log_var))

        self.encoder = Model(inputs=encoder_input, outputs=z, name='encoderVAE')

        # Reconstruction Decoder
        recon_input = Input(shape=(latent_dims,), name='recon_input')
        d1 = Dense(6*6*64, activation='relu')(recon_input)
        d1 = Reshape(target_shape=(6,6,64))(d1)
        d1 = Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu')(d1)
        d1 = Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu')(d1)
        d1 = Conv2DTranspose(16, (3,3), strides=2, padding='same', activation='relu')(d1)
        d1 = Conv2DTranspose(1, (3,3), padding='same', activation='relu', name='recon')(d1)

        self.recon_decoder = Model(inputs=recon_input, outputs=d1, name='reconDecoderVAE')
        
        # Emotion Decoder
        emotion_input = Input(shape=(latent_dims,), name='emotion_input')        

        if linear:
            if sparse:
                d2 = Dense(10, activation='softmax', name='emotion_pred',
                           kernel_regularizer=L1(l1=0.01))(emotion_input)
            else:
                d2 = Dense(10, activation='softmax', name='emotion_pred')(emotion_input)

        else:
            d2 = Dense(64, activation='relu')(emotion_input)
            d2 = Dense(32, activation='relu')(d2)
            d2 = Dense(10, activation='softmax', name='emotion_pred')(d2)
            
        self.emotion_decoder = Model(inputs=emotion_input, outputs=d2, name='emotionDecoderVAE')

        # Define full model
        output_img = self.recon_decoder(z)
        output_emotion = self.emotion_decoder(z)
        self.model = Model(inputs=encoder_input, outputs=[output_img, output_emotion], name='EmotionVAE')
        
        # Add KL loss
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1) * beta
        self.model.add_loss(kl_loss)

        # Compile model
        losses = {'reconDecoderVAE': scaled_mse,
                  'emotionDecoderVAE': categorical_crossentropy}

        loss_weights = {'reconDecoderVAE': 1.0,
                        'emotionDecoderVAE': 1.5}

        metrics = {'emotionDecoderVAE': categorical_accuracy}

        self.model.compile(optimizer=Adam(), loss=losses, loss_weights=loss_weights, metrics=metrics)
        
    def fit(self, inputs, labels, epochs, batch_size, val_data=None, workers=1, callbacks=None):
        history = self.model.fit(inputs, labels, batch_size=batch_size, epochs=epochs,
                                 validation_data=val_data, workers=workers,
                                 callbacks=callbacks)
        
        return history
        
    def get_encoder(self):
        encoder = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('Sampling').output)
        
        return encoder
    
    def get_recon_decoder(self):
        recon_decoder = Model(inputs=self.model.get_layer('reconDecoderVAE').input,
                              outputs=self.model.get_layer('reconDecoderVAE').output)
        
        return recon_decoder

    def get_emotion_decoder(self):
        emotion_decoder = Model(inputs=self.model.get_layer('emotionDecoderVAE').input,
                                outputs=self.model.get_layer('emotionDecoderVAE').output)
        
        return emotion_decoder
        
    def summary(self):
        self.model.summary()
        
    def encode(self, inputs):
        encoder = self.get_encoder()
        encodings = encoder.predict(inputs)
        
        return encodings
    
    def decode(self, inputs):
        recon_decoder = self.get_recon_decoder()
        recon = recon_decoder.predict(inputs)

        emotion_decoder = self.get_emotion_decoder()
        emotion = emotion_decoder.predict(inputs)
        
        return recon, emotion
