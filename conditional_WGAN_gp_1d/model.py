import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import os
import sys


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc_layer_z1 = layers.Dense(128, use_bias=False, activation=tf.nn.leaky_relu)

        self.fc_layer_y1 = layers.Dense(128, use_bias=False, activation=tf.nn.leaky_relu)

        self.combine_dense = layers.Dense(256, use_bias=False, activation=tf.nn.leaky_relu)

        self.fc_layer_2 = layers.Dense(512, use_bias=False, activation=tf.nn.leaky_relu)

        self.output_layer = tf.keras.layers.Dense(72, use_bias=False, activation='sigmoid')

    def call(self, inputs, labels, training=False):
        z = self.fc_layer_z1(inputs)
        y = self.fc_layer_y1(labels)

        # combine_dense
        combined_x = self.combine_dense(tf.concat([z, y], axis=-1))
        x = self.fc_layer_2(combined_x)
        x = self.output_layer(x)

        return x


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc_layer_1 = layers.Dense(512, use_bias=False, activation=tf.nn.leaky_relu)

        self.fc_layer_2 = layers.Dense(256, use_bias=False, activation=tf.nn.leaky_relu)

        self.fc_layer_3 = layers.Dense(128, use_bias=False, activation=tf.nn.leaky_relu)

        self.fc_layer_label = layers.Dense(128, use_bias=False, activation=tf.nn.leaky_relu)

        self.valid_layer = layers.Dense(1, use_bias=False, activation='sigmoid')

        self.output_labels = layers.Dense(1, use_bias=False, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.fc_layer_1(inputs)
        x = self.fc_layer_2(x)
        x = self.fc_layer_3(x)

        # Discriminator
        result = self.valid_layer(x)

        if training:
            # Recognition
            label = self.fc_layer_label(x)
            label = self.output_labels(label)

            return result, label
        else:
            return result
