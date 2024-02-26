#### we referenced code from: https://github.com/ignacio-rocco/cnngeometric_pytorch

import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import scipy.misc
from skimage import io
import pandas as pd
import os
import skimage
from scipy.ndimage import gaussian_filter
tf.keras.backend.set_floatx('float32')

### the output of each cost function is a tensor of shape TensorShape([batch_size])
def training_image_reader(image_names, mask_names, dir, output_shape = (256,256)):

    num_of_images = len(image_names)
    
    image_vol = np.zeros([num_of_images,output_shape[1],output_shape[0], 3])
    mask_vol = np.zeros([num_of_images,output_shape[1],output_shape[0], 1])

    for idx in range(0,num_of_images):
        #print(image_names[idx])
        image_path = os.path.join(dir,image_names[idx])
        
        image_array = cv2.imread(image_path)/255.0
        image_array = cv2.resize(image_array, output_shape)
        image_vol[idx,:,:,:] = image_array
        image_vol = image_vol.astype('float32')
        
        mask_path = os.path.join(dir,mask_names[idx])
        mask_array = cv2.imread(mask_path)[:,:,0]
        mask_array = cv2.resize(mask_array, output_shape)
        mask_vol[idx,:,:,0] = mask_array/255.0
        mask_vol = mask_vol.astype('float32')
        
    dataset = {'image_vol': image_vol, 'mask_vol': mask_vol}
    
    return dataset
