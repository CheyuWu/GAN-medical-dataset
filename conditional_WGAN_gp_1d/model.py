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

        self.fc_layer_3 = layers.Dense(512, use_bias=False, activation=tf.nn.leaky_relu)

        self.output_layer = tf.keras.layers.Dense(62, use_bias=False, activation='sigmoid')

    def call(self, inputs, labels):
        z = self.fc_layer_z1(inputs)
        y = self.fc_layer_y1(labels)

        # combine_dense
        combined_x = self.combine_dense(tf.concat([z, y], axis=-1))
        # x = self.fc_layer_2(combined_x)
        # x = self.fc_layer_3(x)
        x = self.output_layer(combined_x)
        
        return x


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc_layer_z1 = layers.Dense(512, use_bias=False, activation=tf.nn.relu)#512

        self.fc_layer_y1 = layers.Dense(512, use_bias=False, activation=tf.nn.relu)#512

        self.fc_layer_2 = layers.Dense(256, use_bias=False, activation=tf.nn.relu)#256
        
        self.fc_layer_3 = layers.Dense(256, use_bias=False, activation=tf.nn.leaky_relu)#128 256
        
        self.fc_layer_4 = layers.Dense(256, use_bias=False, activation=tf.nn.leaky_relu)#128 256
        
        self.fc_layer_label = layers.Dense(128, use_bias=False, activation=tf.nn.leaky_relu)#128
        
        self.fc_layer_valid = layers.Dense(128, use_bias=False, activation=tf.nn.leaky_relu)

        self.valid_layer = layers.Dense(1, use_bias=False, activation='sigmoid')# sigmoid

        self.output_labels = layers.Dense(1, use_bias=False, activation='softmax')#softmax

    def call(self, inputs, labels, label_out=False):
        z = self.fc_layer_z1(inputs)
        y = self.fc_layer_y1(labels)
        x = self.fc_layer_2(tf.concat([z,y], axis=-1))# combine_dense
        # x = self.fc_layer_3(x)
        # x = self.fc_layer_4(x)
        # Discriminator
        
        # v = self.fc_layer_valid(x)
        result = self.valid_layer(x)
        if label_out:
            # Recognition
            label = self.fc_layer_label(x)
            label = self.output_labels(label)
            
            return result, label
        else:
            return result


