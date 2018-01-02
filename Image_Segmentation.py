# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 12:19:36 2017

@author: sumit gupta
"""
# =============================================================================
# =============================================================================
# ### Convolutional Neural Network ###
# =============================================================================
# =============================================================================

## Part 1: Building the CNN

# Importing keras libraries and packages
from keras.models import Sequential  # Initialises the NN
from keras.layers import Convolution2D # For 2-D images
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step 1: Convolution
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation="relu"))
""" nb_ filter(Filter detector) will be same as no of feature maps as each 
filter will produce one feature map"""

""" We will use ReLU as activation function to remove negative pixel values
in the feature map and have non-linearity in our CNN """

# Step 2: Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

## Additional step to add a second Convolutional layer
""" This step is in addition to the other steps,since we obtained an accuracy
of 75% and wished to see if adding another layer would further improve the 
accuracy """
classifier.add(Convolution2D(32,3,3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Full Connection
classifier.add(Dense(output_dim = 128, activation="relu"))
classifier.add(Dense(output_dim = 1, activation="sigmoid")) # Output layer

# Compiling the CNN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', 
                   metrics= ['accuracy'])

## Part 2: Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                    rescale = 1./255,
                    shear_range= 0.2,
                    zoom_range= 0.2, 
                    horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size= (64,64),
                                                 batch_size= 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size= (64,64),
                                            batch_size= 32,
                                            class_mode = 'binary')


classifier.fit_generator(training_set, 
                         samples_per_epoch = 8000,
                         nb_epoch =25,
                         validation_data= test_set,
                         nb_val_samples = 2000)

# =============================================================================
#  Accuracy with one convolutional layer:
#     Accuracy on train set = 78.94%
#     Accuracy on test set  = 78.46% 
# 
#  Accuracy after adding second convulational layer:
#     Accuracy on train set = 86.64%
#     Accuracy on test set  = 81.76% 
# =============================================================================
   
## Part 3: Making a new Prediction on 2 images to classify them into cat or dog

import numpy as np
from keras.preprocessing import image

# Loading the image to be classified
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',
                            target_size= (64,64))

# Image to array function (2D image to 3D array)
test_image = image.img_to_array(test_image)

# Adding 4th dimension -'batch' for prediction
"""Since predict function in Neural networks accepts inputs in a batch and not 
single inputs."""
test_image = np.expand_dims(test_image, axis=0) # Use numpy for expand_dims
result = classifier.predict(test_image)

# Using class_indices for mapping of 1 or 0 to cats/dogs
training_set.class_indices

# Writing an if-else statement for prediction
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
# Predicting for the second test image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',
                            target_size= (64,64))
test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0) 
result2 = classifier.predict(test_image)
training_set.class_indices

if result2[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'









