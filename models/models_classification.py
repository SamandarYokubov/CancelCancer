import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2



class ClassificationModel(object):

    def __init__(self):
        self.num_classes = 9
        self.img_height = 180
        self.img_width = 180

        with tf. device("cpu:0"):
            self.model = Sequential()
            self.model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)))
            self.model.add(Conv2D(16, 3, padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Dropout(0.25))

            self.model.add(Conv2D(32, 3, padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Dropout(0.25))

            self.model.add(Conv2D(64, 3, padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Dropout(0.25))

            self.model.add(Flatten())
            self.model.add(Dense(128))
            self.model.add(Activation('relu'))
            self.model.add(Dense(self.num_classes))
            self.model.compile(optimizer='adam',
                    loss = SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    def get_model(self):
        with tf. device("cpu:0"):
            self.model.compile(optimizer='adam',
                loss = SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        return self.model





