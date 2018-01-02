# Image Segmentation using CNN
## Dogs Vs Cats
This is the code for a double layered Convolutional Neural network that classifies images into cats and dogs from a pool of 10000 images.
# Overview
I built the model using Keras library, which is built on top of Tensorflow and Theano. The data set consist of 10000 pictures of dogs and cats. The provided images has different sizes. I used adam for stochastic optimization, and binary crossentropy as the loss function.
# Dependencies

● tensorflow
● keras
● numpy
● pandas

# Dataset

The dataset contains 10000 colored images of cats and dogs with different sizes. Two folders are created in the dataset, One for training data which contains 8000 images and other one for test data which contains 2000 images randomly. One another folder containing one image each of a cat and a dog is created for single prediction.
# Usage
Run Image_Segmentation.py in terminal to see the network in training. I have used Spyder from Anaconda to script and visualize the code.


'''

0.8459 - val_loss: 0.4135 - val_acc: 0.8163
Epoch 21/25
250/250 [==============================] - 626s - loss: 0.3417 - acc: 0.8475 - val_loss: 0.4398 - val_acc: 0.8140
Epoch 22/25
250/250 [==============================] - 418s - loss: 0.3257 - acc: 0.8544 - val_loss: 0.4407 - val_acc: 0.8135
Epoch 23/25
250/250 [==============================] - 435s - loss: 0.3178 - acc: 0.8642 - val_loss: 0.4935 - val_acc: 0.7969
Epoch 24/25
250/250 [==============================] - 505s - loss: 0.3102 - acc: 0.8636 - val_loss: 0.4149 - val_acc: 0.8287
Epoch 25/25
250/250 [==============================] - 470s - loss: 0.3032 - acc: 0.8664 - val_loss: 0.4243 - val_acc: 0.8176
'''

