#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:18:50 2022

@author: kauetnd
"""

# Import necessary libraries
from datetime import datetime
import time
from tensorflow import keras 
import tensorflow as tf
import segmentation_models as sm
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from nibabel.affines import rescale_affine
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import nibabel as nib
import pandas as pd
import sys
import pickle

from keras_unet_collection import models  # Importing U-Net model collection

# Print TensorFlow and Keras versions
print(tf.__version__)
print(keras.__version__)

# Check if GPU is available
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Get command-line arguments
# sys.argv[1]: Model architecture
# sys.argv[2]: Type of 2D slices (2DSag, 2DCor, 2DAxi)
# sys.argv[3]: Fold number
# sys.argv[4]: Type of U-Net model

type_of_2d = sys.argv[2]
fold = sys.argv[3]
type_Unet = sys.argv[4]
print(sys.argv[1], type_of_2d, fold, type_Unet)

# Load training and validation datasets
X_train = np.load(f'X_train_{type_of_2d}_FUSFACC_fold{fold}.npy')
X_test = np.load(f'X_val_{type_of_2d}_FUSFACC_fold{fold}.npy')
y_train = np.load(f'y_train_{type_of_2d}_FUSFACC_fold{fold}.npy')
y_test = np.load(f'y_val_{type_of_2d}_FUSFACC_fold{fold}.npy')

# Print dataset sizes
print('sizes')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Define model parameters
architecture = sys.argv[1]  # Model architecture
activation = 'sigmoid'  # Activation function
patch_size = 64  # Size of image patches
n_classes = 1  # Number of classes (binary segmentation)
LR = 1e-4  # Learning rate
batch = 64  # Batch size
steps_per_epoch = X_train.shape[0] / batch  # Compute steps per epoch
save_period = 20  # Frequency of model saving

# Define optimizer and loss functions
optim = keras.optimizers.Adam(LR)  # Adam optimizer

# Define loss functions
# Dice loss helps in handling class imbalance
# Focal loss is useful for handling hard-to-classify pixels
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5])) 
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + focal_loss  # Combined loss function

# Define evaluation metrics
metrics = [sm.metrics.IOUScore(), sm.metrics.FScore()]

# Print current time
current_time = time.strftime("%H:%M:%S", time.localtime())
print(current_time)

# Define model checkpoint path
checkpoint_path = f'{type_Unet}/fold{fold}/{architecture}_{type_of_2d}'

# Define model checkpoint callback
callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path + '-{iou_score:.2f}',
    verbose=1,
    save_best_only=True,
    monitor='iou_score',
    mode='max'
)

# Select and initialize the U-Net model based on user input
if type_Unet == 'Attention_Unet':
    model = models.att_unet_2d(
        (patch_size, patch_size, 1),
        filter_num=[64, 128, 256, 512, 1024],
        n_labels=1,
        stack_num_down=2,
        stack_num_up=2,
        activation='ReLU',
        atten_activation='ReLU',
        attention='add',
        output_activation='Sigmoid',
        batch_norm=True,
        pool=False,
        unpool=False,
        backbone='VGG16',
        weights=None,
        freeze_backbone=False,
        freeze_batch_norm=False,
        name='attunet'
    )
elif type_Unet == 'Unet2p':
    model = models.unet_plus_2d(
        (patch_size, patch_size, 1),
        filter_num=[64, 128, 256, 512, 1024],
        n_labels=1,
        stack_num_down=2,
        stack_num_up=2,
        activation='ReLU',
        output_activation='Sigmoid',
        batch_norm=True,
        pool=False,
        unpool=False,
        deep_supervision=False,
        backbone='VGG16',
        weights=None,
        freeze_backbone=False,
        freeze_batch_norm=False,
        name='xnet'
    )
elif type_Unet == 'Unet3p':
    model = models.unet_3plus_2d(
        (patch_size, patch_size, 1),
        n_labels=1,
        filter_num_down=[64, 128, 256, 512, 1024],
        filter_num_skip='auto',
        filter_num_aggregate='auto',
        stack_num_down=2,
        stack_num_up=1,
        activation='ReLU',
        output_activation='Sigmoid',
        batch_norm=True,
        pool=False,
        unpool=False,
        deep_supervision=False,
        backbone='VGG16',
        weights=None,
        freeze_backbone=False,
        freeze_batch_norm=False,
        name='unet3plus'
    )
elif type_Unet == 'Unet':
    model = sm.Unet(
        architecture,
        classes=1,
        input_shape=(patch_size, patch_size, 1),
        encoder_weights=None,
        activation=activation
    )

# Compile the model
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

# Print model summary
print(model.summary())

# Set initial training epoch
initial_epoch = 0

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=batch,
    steps_per_epoch=steps_per_epoch,
    initial_epoch=initial_epoch,
    callbacks=[callback],
    epochs=600,
    verbose=1,
    validation_data=(X_test, y_test)
)

# Print final time
current_time = time.strftime("%H:%M:%S", time.localtime())
print(current_time)
