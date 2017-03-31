import pickle
import numpy as np
import math
from sklearn.utils import shuffle

import tensorflow as tf
print('tensorflow: ', tf.__version__)

from keras import __version__ as keras_version
print('keras: ', keras_version)
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.activations import elu, relu, softmax
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Load the data


# Preprocess the data


# Build the model


# Train the model


# Tune parameters and optimize the model


# Test the results


if __name__ == '__main__':
	try:
		# TODO: train the model

		# TODO: save the model
		filepath = './model.h5'
		model.save(filepath)

	except:
		# TODO: uh oh
