from __future__ import print_function

import os
import sys
from argparse import ArgumentParser
from time import time

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc
from tensorflow.keras.optimizers import Adam

### import self-defined functions
from UNet import *

from image_reader import *
from loss import *
import SimpleITK as sitk

tf.keras.backend.set_floatx('float32')


def build_parser():
    parser = ArgumentParser()
    
    # Paths, change here please
    parser.add_argument('--image-dir', type=str, default='../preprocessing/', help='path to folder of training images')
    parser.add_argument('--csv-file', type=str, default='../training_data/train.csv', help='path to csv file of training examples')
    parser.add_argument('--validation-csv-file', type=str, default='../training_data/val.csv', help='path to csv file of validation examples')
    parser.add_argument('--trained-model-dir', type=str, default='../trained_models/', help='path to trained models folder')
    parser.add_argument('--trained-model-fn', type=str, default='weed_detector_batch_size_16_image_size_256_start_neurons_32', help='trained model filename')
    parser.add_argument('--result-name', type=str, default='../trained_models/weed_detector_batch_size_32_image_size_256_start_neurons_32_loss.csv', help='directory to store segmentation results')
    # Optimization parameters 
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num-epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size, 4/8/16/32/64')
    parser.add_argument('--gpu-id', type=int, default=0, help='which gpu to use')
    parser.add_argument('--image-size', type=int, default=256, help='image height, 160/320')
    parser.add_argument('--start-neurons', type=int, default=32, help='start_neurons，16/32/64')
    
    return parser
   
   
def main():
    parser = build_parser()
    args = parser.parse_args()
    
    devices = tf.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.set_visible_devices(devices[args.gpu_id], 'GPU')
    
    train_losses = np.zeros(args.num_epochs)
    validation_losses = np.zeros(args.num_epochs)
    
    data = pd.read_csv(args.csv_file)
    data_validation = pd.read_csv(args.validation_csv_file)
    
    num_of_train_images = data.shape[0]
    num_of_validation_images = data_validation.shape[0]
    

    best_loss = 100000000
    
    image_size = args.image_size
    
    all_image_names = data.values[:,0]
    all_mask_names = data.values[:,1]
    
    all_image_names_validation = data_validation.values[:,0]
    all_mask_names_validation = data_validation.values[:,1]


    #if (len(args.initial_model)>0):
    #    model =  tf.keras.models.load_model(args.initial_model)
    #else:
    
    model = UNet(input_shape = (image_size,image_size,3), start_neurons = args.start_neurons)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    
    print(model.summary())
    

    for epoch in range(1,args.num_epochs+1):
    
        #optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr*np.power(0.92,epoch))
    
        num_of_batches = int(num_of_train_images/args.batch_size)
        s = 0
            
        for idx in range(0,num_of_batches):
            batch_idx = np.random.randint(num_of_train_images, size=args.batch_size)
            
            image_names = all_image_names[batch_idx]
            #print(image_names)
            mask_names = all_mask_names[batch_idx]
            
            dataset = training_image_reader(image_names, mask_names, args.image_dir, output_shape = (image_size,image_size))
  
            vol_train = dataset['image_vol']
            mask_train = dataset['mask_vol']

            ####end of reading training dataset

            with tf.GradientTape(persistent=True) as tape:
                    
                mask_pred = model(vol_train)
                loss = BCE(mask_train, mask_pred)
                #loss = dice_loss(mask_train, mask_pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            
            ### sum up training loss
            s = s + loss.numpy()
            
        ### compute validationing loss
        
        loss_validation = 0
        
        for idx in range(num_of_validation_images):
            dataset_validation = training_image_reader(all_image_names_validation[idx:idx+1], all_mask_names_validation[idx:idx+1], args.image_dir, output_shape = (image_size,image_size))
          

            vol_validation = dataset_validation['image_vol']
            mask_validation = dataset_validation['mask_vol']
     
            mask_pred = model(vol_validation)
            
            #if idx%10==0:
            #### output_predicted mask_names
                #mask_pred_array = np.squeeze(mask_pred.numpy())
                #cv2.imwrite("../evaluation/temp/pred_" + str(idx) + ".png", 255*mask_pred_array )
            
                #mask_validation_array = np.squeeze(mask_validation)
                #cv2.imwrite("../evaluation/temp/true_" + str(idx) + ".png", 255*mask_validation_array )
                
                #vol_validation_array = np.squeeze(vol_validation)
                #cv2.imwrite("../evaluation/temp/image_" + str(idx) + ".png", 255*vol_validation_array )
            ###
            
            #print(tf.math.reduce_max(mask_validation))
            #print(tf.math.reduce_max(mask_pred))

            # mask_array = np.squeeze(mask_pred.numpy())

            # cv2.imwrite('../predicted_mask.png', 255*mask_array)
            
            loss_validation = loss_validation + BCE(mask_validation, mask_pred)
            
        
        loss_validation = loss_validation/num_of_validation_images
        
        if loss_validation < best_loss:
            best_loss = loss_validation
            model.save(args.trained_model_dir + args.trained_model_fn + '_best_loss' + '.h5')
        
        
        print("epoch= " + str(epoch) + ",  train loss = " + str(s/num_of_batches) + ",  validation loss = " + str(loss_validation))
            
        train_losses[epoch-1] = s/num_of_batches
        validation_losses[epoch-1] = loss_validation
            
        
    #save model for each image resolution
    model.save(args.trained_model_dir + args.trained_model_fn +  '.h5')
        
    
    print('done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    array = np.empty((args.num_epochs + 10,3), dtype='U25')
    
    array[0,0] = "epoch"
    array[0,1] = "train_loss"
    array[0,2] = "validation_loss"
    
    for j in range(0,args.num_epochs):
        array[1 + j , 0] = str(j+1)
        array[1 + j , 1] = str(train_losses[j])
        array[1 + j , 2] = str(validation_losses[j])
    np.savetxt(args.result_name, array, delimiter=",", fmt='%s')
    
    
    
if __name__ == '__main__':
    main()
