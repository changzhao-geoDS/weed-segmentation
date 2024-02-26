# reference: https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/ 

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model

tf.keras.backend.set_floatx('float32')

def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x


def UNet(input_shape = (256,256,3), start_neurons = 32, drop_rate = 0.25):
   inputs = layers.Input(input_shape)
   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, start_neurons*1)
   # 2 - downsample
   f2, p2 = downsample_block(p1, start_neurons*2)
   # 3 - downsample
   f3, p3 = downsample_block(p2, start_neurons*4)
   # 4 - downsample
   f4, p4 = downsample_block(p3, start_neurons*8)
   # 5 - bottleneck
   bottleneck = double_conv_block(p4, start_neurons*16)
   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, start_neurons*8)
   # 7 - upsample
   u7 = upsample_block(u6, f3, start_neurons*4)
   # 8 - upsample
   u8 = upsample_block(u7, f2, start_neurons*2)
   # 9 - upsample
   u9 = upsample_block(u8, f1, start_neurons*1)
   # outputs
   outputs = Conv2D(filters=1, kernel_size=1, activation="sigmoid", padding="same")(u9)
   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
   return unet_model