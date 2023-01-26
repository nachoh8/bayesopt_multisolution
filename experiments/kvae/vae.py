'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        self.dim = kwargs.pop('dim')
        self.z_mean = kwargs.pop('z_mean')
        self.z_log_var = kwargs.pop('z_log_var')
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = self.dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x


class VAE(object):

    def __init__(self, outlier_ratio=0):
        self.outlier_ratio = outlier_ratio
        self.x_train, self.x_test = self.preprocess_data()
        self.batch_size = 100
        self.original_dim = 784
        self.latent_dim = 2
        self.intermediate_dim = 256
        self.epochs = 10
        self.lr = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-08
        self.decay = 0.0
        self.intermediate_dim = 256


    def preprocess_data(self):
        # train the VAE on MNIST digits
        (x_train, _), (x_test, _) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        return x_train, x_test

    def sampling(self, args):
        epsilon_std = 1.0
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def define_model(self):
        x = Input(shape=(self.original_dim,))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        y = CustomVariationalLayer(
            dim=self.original_dim,
            z_mean=z_mean,
            z_log_var=z_log_var,
        )([x, x_decoded_mean])
        return Model(x, y)

    def call(self, params):
        self.lr = params[0][0]
        #beta_1 = params[0][1]
        #beta_2 = params[0][2]
        self.epsilon = params[0][1]
        self.decay = params[0][2]
        self.intermediate_dim = int(params[0][3])
        print params
        return self.run_model()

    def run_model(self):
        vae = self.define_model()
        vae.compile(optimizer=Adam(self.lr, self.beta_1, self.beta_2, self.epsilon, self.decay), loss=None)

        if np.random.random() < self.outlier_ratio:
            nimages = np.random.randint(1, 10) * 100
            x_train = self.x_train[1:nimages, :]
        else:
            x_train = self.x_train

        vae.fit(x_train,
                shuffle=True,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(self.x_test, None))

        performance = vae.evaluate(self.x_test, None)
        K.clear_session()
        
        return performance

if __name__ == '__main__':
    vae_model = VAE()
    print "Result:", vae_model.run_model(decay=0.5)
    print "Result:", vae_model.run_model(intermediate_dim=10)

# build a model to project inputs on the latent space
# encoder = Model(x, z_mean)

# # display a 2D plot of the digit classes in the latent space
# x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# plt.colorbar()
# plt.show()

# # build a digit generator that can sample from the learned distribution
# decoder_input = Input(shape=(latent_dim,))
# _h_decoded = decoder_h(decoder_input)
# _x_decoded_mean = decoder_mean(_h_decoded)
# generator = Model(decoder_input, _x_decoded_mean)

# # display a 2D manifold of the digits
# n = 15  # figure with 15x15 digits
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# # to produce values of the latent variables z, since the prior of the latent space is Gaussian
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = generator.predict(z_sample)
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[i * digit_size: (i + 1) * digit_size,
#                j * digit_size: (j + 1) * digit_size] = digit

# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
# plt.show()
