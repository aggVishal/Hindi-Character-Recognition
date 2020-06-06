# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:58:23 2018

@author: Sam
"""
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation,Dropout


# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Convolution2D(100, 5, 5, border_mode='valid', input_shape=(32, 32, 3)))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('tanh'))

# Adding a second convolutional layer
model.add(Convolution2D(250, 5, 5, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('tanh'))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(1000))
# Using dropout to prevent overftting
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(46))
model.add(Activation('softmax'))

# Compiling the CNN
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

#Data Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#Normalization
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Dataset/train',
                                                 target_size = (32, 32),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size = (32, 32),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# for training model
"""model.fit_generator(training_set,
                    nb_epoch=20,
                    validation_data = test_set,
                    nb_val_samples = 2000)
"""

model.summary()
model.save("HCRModel.h5")

#validation accuracy - 95.4%



"""
import os
Labels=os.listdir('dataset/test')

for subdir in Labels:
    print (re.findall('\d+', subdir )[0])
    #print(int(list(filter(str.isdigit, subdir))[0]))
    #print([int(s) for s in subdir if s.isdigit()])
    print(subdir)
"""
