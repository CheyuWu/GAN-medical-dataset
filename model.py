import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import os, sys

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc_layer_1 = layers.Dense(128, use_bias=False, activation='tanh')
#         self.bn_1 = layers.BatchNormalization()

        self.fc_layer_2 = layers.Dense(256, use_bias=False, activation='tanh')
#         self.bn_2 = layers.BatchNormalization()

        self.fc_layer_3 = layers.Dense(512, use_bias=False, activation='tanh')
#         self.bn_3 = layers.BatchNormalization()

        self.output_layer = tf.keras.layers.Dense(73, use_bias, activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.fc_layer_1(inputs)
        x = self.fc_layer_2(x)
        x = self.fc_layer_3(x)
        x = self.output_layer(x)
        
        return x
        
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc_layer_1 = layers.Dense(512, use_bias=False, activation='tanh')
#         self.bn_1 = layers.BatchNormalization()

        self.fc_layer_2 = layers.Dense(256, use_bias=False, activation='tanh')
#         self.bn_2 = layers.BatchNormalization()

        self.fc_layer_3 = layers.Dense(128, use_bias=False, activation='tanh')

#         self.bn_3 = layers.BatchNormalization()
        self.output_layer = layers.Dense(1, use_bias=False, activation='linear')

    def call(self, inputs, training=None):
        x = fc_layer_1(inputs)
        x = self.fc_layer_2(x)
        x = self.fc_layer_3(x)
        x = self.output_layer(x)

        return x