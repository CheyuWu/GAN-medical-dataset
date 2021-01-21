from all_funcs import util
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import os
import sys
sys.path.append("..")


class Generator(keras.Model):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.combine_dense = layers.Dense(
            62, use_bias=False, activation=tf.nn.tanh)

        self.fc_layer_1 = layers.Dense(
            31, use_bias=False, activation=tf.nn.tanh)

        self.fc_layer_2 = layers.Dense(
            62, use_bias=False, activation=tf.nn.tanh)

        self.output_layer = tf.keras.layers.Dense(
            latent_dim+1, use_bias=False, activation='sigmoid')

    def call(self, noise, org_data, labels):
        # combine_dense
        combined_x = self.combine_dense(
            tf.concat([org_data, noise, labels], axis=-1))
        x = self.fc_layer_1(combined_x)
        x = self.fc_layer_2(x)
        x = self.output_layer(x)
        return x


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc_layer_1 = layers.Dense(
            63, use_bias=False, activation=tf.nn.tanh)

        self.fc_layer_2 = layers.Dense(
            32, use_bias=False, activation=tf.nn.tanh)  # 256

        self.fc_layer_3 = layers.Dense(
            63, use_bias=False, activation=tf.nn.tanh)  # 128 256

        self.output_layer = layers.Dense(
            1, use_bias=False, activation='linear')  # softmax, sigmoid

    def call(self, inputs):
        x = self.fc_layer_1(inputs)
        x = self.fc_layer_2(x)
        x = self.fc_layer_3(x)
        x = self.output_layer(x)
        return x


@tf.function
def train_discriminator(x, label, generator, discriminator, dis_optimizer, latent_dim=62, ):

    noise = tf.random.normal([x.shape[0], latent_dim])

    with tf.GradientTape() as dis_tape:
        gen_data = generator(noise, x, label)
        dis_output = discriminator(gen_data)
        ## concat real_data and label
        real_data_with_label = tf.concat([x, label], axis=1)
        real_output = discriminator(real_data_with_label)

        # formula of Gradient penalty
        x_hat = util.random_weight_average(real_data_with_label, gen_data)
        d_hat = discriminator(x_hat)

        disc_loss = util.discriminator_loss(
            real_output, dis_output, d_hat, x_hat)

    grad_disc = dis_tape.gradient(disc_loss, discriminator.trainable_variables)
    dis_optimizer.apply_gradients(
        zip(grad_disc, discriminator.trainable_variables))

    return disc_loss


@tf.function
def train_generator(org_data, label, generator, discriminator, gen_optimizer, params, batch_size=128, latent_dim=62):
    
    ## To calculate condition is right or wrong
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    ## create noise
    noise = tf.random.normal([batch_size, latent_dim])
    
    with tf.GradientTape() as gen_tape:
        gen_data = generator(noise, org_data, label)
        dis_output = discriminator(gen_data)

        gen_loss = util.generator_loss(dis_output)
        
        ## sum all loss
        sum_loss = gen_loss + \
            util.identifiability(gen_data, org_data) + \
            util.reality_constraint(gen_data, params)+ \
            cross_entropy(label, gen_data[:,-1])

    grad_gen = gen_tape.gradient(sum_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

    return sum_loss
