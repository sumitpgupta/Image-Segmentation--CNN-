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
Run ```Image_Segmentation.py``` in terminal to see the network in training. I have used Spyder from Anaconda to script and visualize the code.
# Test Run

## Predicting a new single case

```
# Using class_indices for mapping of 1 or 0 to cats/dogs
training_set.class_indices

# Writing an if-else statement for prediction
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
 ```


```
API: `fit_generator(<keras.pre..., validation_data=<keras.pre..., steps_per_epoch=250, epochs=25, validation_steps=2000)`
Epoch 1/25
250/250 [==============================] - 430s - loss: 0.6684 - acc: 0.5830 - val_loss: 0.6225 - val_acc: 0.6726
Epoch 2/25
250/250 [==============================] - 391s - loss: 0.5980 - acc: 0.6813 - val_loss: 0.5594 - val_acc: 0.7146
Epoch 3/25
250/250 [==============================] - 522s - loss: 0.5591 - acc: 0.7151 - val_loss: 0.5699 - val_acc: 0.7005
Epoch 4/25
250/250 [==============================] - 394s - loss: 0.5321 - acc: 0.7303 - val_loss: 0.5269 - val_acc: 0.7437
Epoch 5/25
250/250 [==============================] - 378s - loss: 0.5136 - acc: 0.7400 - val_loss: 0.4964 - val_acc: 0.7655
Epoch 6/25
250/250 [==============================] - 370s - loss: 0.4907 - acc: 0.7611 - val_loss: 0.4891 - val_acc: 0.7742
Epoch 7/25
250/250 [==============================] - 384s - loss: 0.4732 - acc: 0.7724 - val_loss: 0.4757 - val_acc: 0.7656
Epoch 8/25
250/250 [==============================] - 370s - loss: 0.4655 - acc: 0.7805 - val_loss: 0.4574 - val_acc: 0.7813
Epoch 9/25
250/250 [==============================] - 366s - loss: 0.4541 - acc: 0.7865 - val_loss: 0.4448 - val_acc: 0.8024
Epoch 10/25
250/250 [==============================] - 391s - loss: 0.4368 - acc: 0.7928 - val_loss: 0.4656 - val_acc: 0.7975
Epoch 11/25
250/250 [==============================] - 387s - loss: 0.4297 - acc: 0.7995 - val_loss: 0.4784 - val_acc: 0.7789
Epoch 12/25
250/250 [==============================] - 372s - loss: 0.4191 - acc: 0.8051 - val_loss: 0.4446 - val_acc: 0.8034
Epoch 13/25
250/250 [==============================] - 356s - loss: 0.4078 - acc: 0.8082 - val_loss: 0.4301 - val_acc: 0.8066
Epoch 14/25
250/250 [==============================] - 381s - loss: 0.3903 - acc: 0.8237 - val_loss: 0.4425 - val_acc: 0.8078
Epoch 15/25
250/250 [==============================] - 380s - loss: 0.3939 - acc: 0.8200 - val_loss: 0.4177 - val_acc: 0.8241
Epoch 16/25
250/250 [==============================] - 369s - loss: 0.3766 - acc: 0.8301 - val_loss: 0.4244 - val_acc: 0.8154
Epoch 17/25
250/250 [==============================] - 360s - loss: 0.3607 - acc: 0.8394 - val_loss: 0.4625 - val_acc: 0.8040
Epoch 18/25
250/250 [==============================] - 360s - loss: 0.3572 - acc: 0.8365 - val_loss: 0.4137 - val_acc: 0.8225
Epoch 19/25
250/250 [==============================] - 437s - loss: 0.3534 - acc: 0.8433 - val_loss: 0.4090 - val_acc: 0.8254
Epoch 20/25
250/250 [==============================] - 495s - loss: 0.3429 - acc: 0.8459 - val_loss: 0.4135 - val_acc: 0.8163
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
```

## 
Accuracy: 81.76% on test set
