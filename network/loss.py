#### we referenced code from: https://github.com/voxelmorph/voxelmorph


import tensorflow as tf
import keras.backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc
tf.keras.backend.set_floatx('float32')

def dice_loss(I, J):
    # assumes I, J are sized [batch_size, height, width]
    numerator = 2 * tf.reduce_sum(I * J, [1,2])
    denominator = tf.maximum(tf.reduce_sum(I + J, [1,2]), 1e-5)
    dice = numerator/denominator
    return -tf.reduce_sum(dice)/I.shape[0]


### the output of each cost function is a tensor of shape TensorShape([batch_size])
def BCE(y_true, y_pred):

    batch_size = y_true.shape[0]
    number_of_pixels = y_true.shape[0]*y_true.shape[1]*y_true.shape[2]

    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    
    epsilon = 0.00001
    
    beta = 1.0 - tf.cast(tf.math.count_nonzero(y_true), tf.float32)/tf.cast(tf.size(y_true),tf.float32)
    
    
    loss = -beta*tf.math.reduce_sum(tf.math.multiply(y_true,tf.math.log(y_pred + epsilon))) - (1.0 - beta)*tf.math.reduce_sum(tf.math.multiply(1.0-y_true,tf.math.log(1.0 - y_pred + epsilon)))

    return loss/(number_of_pixels)
