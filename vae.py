# Disable TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, UpSampling2D, Reshape, Conv2DTranspose
from tensorflow.keras.backend import flatten as tf_flat
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model, Input
from glob import glob
import numpy as np
import time

from data_util import *

DATASET_SIZE = 202599 - 2599

# Training
BATCH_SIZE = 100
EPOCHS = 100000
PER_EPOCH = (DATASET_SIZE // BATCH_SIZE) // 2

# Dimensions
KERNEL_SIZE = 3
N_LATENT = 256  # Latent-space dimensions to be used to generate new images


class CVAE:
    def __init__(self):
        # Create the model's input layer, which will define the encoder and decoder accordingly.
        inputs = Input(shape=(OUT_SIZE, OUT_SIZE, CHANNELS))
        self.create_encoder(inputs)
        self.create_decoder()
        outputs = self.decoder(self.encoder(inputs)[2])

        # Creating the complete Variational Autoencoder from the encoder and decoder networks
        self.vae = Model(inputs, outputs, name='vae_model')

        # Add the model's custom loss function
        loss_func = self.compute_loss()
        self.vae.compile(optimizer='rmsprop', loss=loss_func)

        # save the model architecture
        # plot_model(self.vae, to_file='vae_cnn.png', show_shapes=True)

    def create_encoder(self, inputs):
        """
        Encoder Network - Q(z|X)
        Input: inputs - Input layer for the model, defining the network's input shape and the way
                        the data is processed through the network.
        """
        x = inputs
        conv_filters = 120
        for i in range(4):
            x = Conv2D(filters=conv_filters, kernel_size=KERNEL_SIZE,
                       padding="same", activation="relu")(x)
            x = MaxPool2D(padding="same")(x)
            conv_filters += 40

        # Saving the current shape of the data as refrence to create a symmetrical decoder
        self.decode_shape = x.shape[1:]
        x = Flatten()(x)
        x = Dense(units=N_LATENT/4)(x)

        # Calculating the Mean (mu) and Standard deviation (log_sigma), which will be used
        # to calculate the KL divergence loss and enforce unit Gaussian distribution
        self.mu = Dense(units=N_LATENT)(x)
        self.log_sigma = Dense(units=N_LATENT)(x)

        # The latent vector, z, is modified to create new data points
        z = self.reparameterize(self.mu, self.log_sigma)

        # Saving the encoder as a model
        self.encoder = Model(
            inputs, [self.mu, self.log_sigma, z], name='encoder')
        # plot_model(self.encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    def create_decoder(self):
        """
        Decoder network - P(X|z)
        """
        latent_inputs = Input(shape=(N_LATENT,))
        x = Dense(units=np.prod(self.decode_shape))(latent_inputs)
        x = Reshape(self.decode_shape)(x)

        conv_filter = 200
        for i in range(3):
            x = Conv2DTranspose(filters=conv_filter, kernel_size=KERNEL_SIZE,
                                strides=2, padding="same", activation="relu")(x)
            conv_filter -= 40

        x = Conv2DTranspose(filters=CHANNELS, kernel_size=KERNEL_SIZE,
                            strides=2, padding="same", activation="sigmoid")(x)

        img = Reshape((-1, OUT_SIZE, OUT_SIZE, CHANNELS))(x)
        self.decoder = Model(latent_inputs, x, name='decoder')
        # plot_model(self.decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    def reparameterize(self, mu, log_sigma):
        """
        Using the reparameterization trick to sample from the latent space
        Inputs: mu - the encoded data's mean and location in the latent space
                log_sigma - A standard deviation to alter the the encoded data to represent a new image
        Outputs: The sampled latent vector
        """
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(log_sigma / 2) * eps

    def compute_loss(self):
        """
        Custom loss function. In order to train a model, the loss function must receive only two parameters - y_true and y_predict.
        In order to use class members (self.log_sigma, self.mu), a wrapper function is used to bypass this limit.
        Inputs: inputs - original image to feed to the model
                outputs - the image recreated by the model
        Output: Total loss for a given image
        """
        def calc_loss(inputs, outputs):
            reconstruction_loss = tf.metrics.binary_crossentropy(
                tf_flat(inputs), tf_flat(outputs))
            reconstruction_loss *= OUT_SIZE * OUT_SIZE
            kl_loss = -0.5 * tf.reduce_sum(1.0 + self.log_sigma - tf.square(
                self.mu) - tf.exp(self.log_sigma), 1)
            return tf.reduce_mean(reconstruction_loss + kl_loss)
        return calc_loss

    def train(self, dataset):
        """
        Train the VAE model
        """
        train_loss = []
        test_images = np.array([get_image(dataset[i], OUT_SIZE, CHANNELS)
                                for i in range(64)]).astype(np.float32)

        start = time.time()
        print("[+] Starting Training")

        for iters in range(EPOCHS):
            history = self.vae.fit_generator(
                prepare_epoch(dataset), steps_per_epoch=PER_EPOCH, epochs=1)

            loss = history.history['loss'][-1]
            train_loss.append(loss)
            print("Epoch {} Loss: {}\n[+] Time since start: {}".format(
                iters, str(loss), time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))
            plotScores(train_loss, [], "EncoderScores.png")

            # Every X iterations, save & test the model
            if iters % 1 == 0:
                self.vae.save_weights(WEIGHTS)

                # Example tests
                faces = self.vae.predict(test_images)
                imsave(faces, [8, 8], "./tests/test{}.png".format(iters))
                # make_gif()
                print("[+] Saved")


def prepare_epoch(dataset):
    """
    Processing the dataset using numpy, and sorting it into batches every epoch
    """
    print("[-] Epoch Start")

    i = 0
    for sample in range(len(dataset)):
        if sample <= i + BATCH_SIZE-1:
            continue

        batch = []
        for i in range(i, i+BATCH_SIZE):
            batch.append(get_image(dataset[i], OUT_SIZE, CHANNELS))

        i += BATCH_SIZE + 1

        batch_images = np.array(batch).astype(np.float32)
        yield (batch_images, batch_images)
    print("i: {}, s: {}".format(i, sample))

    print("[+] Epoch complete")


if __name__ == "__main__":
    print("[+] Starting training sequence...")
    # Check if tensorflow recognises the GPU as well as the CPU
    from tensorflow.python.client import device_lib

    # Allow GPU memory growth to prevent crashes
    tf.config.gpu.set_per_process_memory_growth(True)

    if "GPU" in device_lib.list_local_devices():
        print("[!] Error: Unable to train model - no GPU found")
        exit()

    net = CVAE()

    dataset = glob(PATH)
    dataset = sorted(dataset)
    dataset = np.array(dataset)

    net.train(dataset)
