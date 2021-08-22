import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
from kt_utils import *

# 使用抽象keras backend API，可以兼容多种backend
# Keras技术文档中对backend称为后端，处理如张量乘积和卷积等低级操作
# Keras以模块的方式将几个不同的后端引擎嵌入到Keras中
# Keras主要有三个后端可用：TensorFlow、Theano、CNTK
import tensorflow.keras.backend as K

# Image_data_format()，返回默认图像的维度顺序(“channels_first"或"channels_last”)。
# 彩色图像的性质一般包括：width、height、channels
# 以普通的256*256*3RGB图像为例：
# 选择channels_first：返回(3,256,256)
# 选择channels_last：返回(256,256,3)
# 默认设置为channels_last
K.set_image_data_format('channels_last')


import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import cv2

#%%Dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


#%% GRADED FUNCTION: HappyModel
def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
        
    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape. 
    # Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes   
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X   
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model = Model(inputs=X_input, outputs=X, name='HappyModel')
        
    return model

#%% Step 1: create the model
happyModel = HappyModel(X_train.shape[1:])

#%% Step 2: compile the model
# See https://keras.io/optimizers/
happyModel.compile("adam","binary_crossentropy", metrics=['accuracy'])

#%% Step 3: train the model
# If you run fit() again, the model will continue to train with the parameters
# it has already learned instead of reinitializing them.
happyModel.fit(X_train, Y_train, epochs=40, batch_size=50)

#%% Step 4: evaluate model
preds = happyModel.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#%% Test my own image

img_path = 'images/my_image.jpg'

img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))
#%% Summary
happyModel.summary()
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))






