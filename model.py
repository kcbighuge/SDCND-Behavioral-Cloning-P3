import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import csv
import cv2

import tensorflow as tf
print('tensorflow: ', tf.__version__)

from keras import __version__ as keras_version
print('keras: ', keras_version)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Cropping2D
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Convolution2D
print('Modules loaded.')

# Create a generator function for training the network
def generator(samples, batch_size=32):
    '''Generate image input features and steering angle target
    Camera positions: center 0, left 1, right 2
    '''
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])   
                
                # Get the left camera images
                name = '../sample-data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)  # load img as BGR        
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)                
                images.append(left_image)
                angles.append(center_angle+0.25)  # adjust to steer right
                
                # Get the right camera images
                name = '../sample-data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)  # load img as BGR        
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)                
                images.append(right_image)
                angles.append(center_angle-0.25)  # adjust to steer left
    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def train_model():
	'''Create the network, train, and return.'''
	# Load the data
	samples = []
	with open('../sample-data/driving_log.csv') as csvfile:
	    reader = csv.reader(csvfile)
	    for line in reader:
	        samples.append(line)

	# Remove header row
	if type(samples[0][0]) == str:
	    samples = samples[1:] 

	# Set up training & validation sets
	train_samples, valid_samples = train_test_split(samples, test_size=0.25)

	# Compile and train the model using the generator function
	train_generator = generator(train_samples, batch_size=32)
	valid_generator = generator(valid_samples, batch_size=32)

	# Input immage dimensions
	ch, row, col = 3, 160, 320

	# Create a model
	model = Sequential()

	# Preprocess incoming data, centered around zero with small standard deviation 
	model.add(Lambda(lambda x: x/127.5 - 1.,
	        input_shape=(row, col, ch), 
	        output_shape=(row, col, ch)))

	# Crop row pixels from top, bottom
	CROP_TOP = 50
	CROP_BOTTOM = 30
	model.add(Cropping2D(cropping=((CROP_TOP,CROP_BOTTOM), (0,0)), input_shape=(row, col, ch)))

	# Build the model from CommaAI
	# https://github.com/commaai/research/blob/master/train_steering_model.py
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())

	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	# Train the model
	EPOCH = 10
	model.compile(loss='mse', optimizer='adam')
	early_stop = EarlyStopping(monitor='val_loss', patience=1)
	model.fit_generator(train_generator, samples_per_epoch=2*len(train_samples), callbacks=[early_stop], 
		validation_data=valid_generator, nb_val_samples=len(valid_samples), nb_epoch=EPOCH)

	return model


if __name__ == '__main__':
	try:
		# Train the network
		model = train_model()

		# Save the model
		filepath = './model.h5'
		model.save(filepath)
		print('Model saved.')

	except:
		print('\nAn error occured :(')
