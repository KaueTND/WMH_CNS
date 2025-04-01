#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:18:50 2022

@author: kauetnd
"""

# Import standard libraries and modules
from datetime import datetime
import time

# Import TensorFlow and Keras libraries
from tensorflow import keras
import tensorflow as tf

# Import segmentation models for 3D segmentation
import segmentation_models_3D as sm

# Import image processing and utility libraries
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from nibabel.affines import rescale_affine
from matplotlib import pyplot as plt

# Import Keras utilities for model loading and backend operations
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

# Import scikit-learn utilities for data splitting and K-Fold cross-validation
from sklearn.model_selection import train_test_split, KFold

# Import nibabel for handling NIfTI image files
import nibabel as nib

# Import pandas and other utility libraries
import pandas as pd
import sys
import pickle

# Read command-line arguments to set parameters for model and data slicing
architecture = sys.argv[1]
type_technique = sys.argv[2]  # e.g., 'Unet'
type_slicing = sys.argv[3]    # e.g., '2DAxi'

# Print the received command-line arguments
print(architecture, type_technique, type_slicing)
# Example comment: LinkNet 2DAxi

# Print the TensorFlow and Keras versions for debugging purposes
print(tf.__version__)
print(keras.__version__)

# Check if a GPU device is available for computation
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))  # This line will not execute if GPU is not found

# Indicate the start of dataset loading and affine reading
print('1) read affines and load *total_6dataset')

# Load the total image and mask datasets from .npy files
img_total = np.load('img_total_FUSFACC.npy')
mask_total = np.load('mask_total_FUSFACC.npy')
print(img_total.shape)
print(mask_total.shape)

# Assign loaded datasets to new variables (these assignments appear redundant)
vec_img = input_img = img_total
vec_mask = input_mask = mask_total

# Load the affine transformation matrices
affines = np.load('affines_FUSFACC.npy')

# Load the list of participant file names from a text file
files = list()
with open('participants_FUSFACC.txt', 'r') as f:
    files.append(f.read())
files = np.array(files)
files = files[0].split('\n')
print(files)

# Begin identifying the important patches from the mask dataset
print('read position_important_patches')
print(img_total.shape)
positions_important_patches = list()
print(mask_total.shape)  # Expecting mask shape: n_patches, x, y, z

# Initialize counter for patches meeting criteria
count = 0
for idx, shape in enumerate(mask_total):
    for i in range(64):
        # Check if the unique values in the specified slice equal 2
        if type_slicing == '2DAxi':
            if (np.unique(shape[:,:,i].ravel()).shape[0] == 2): 
                count = count + 1
                positions_important_patches.append([idx, i])
        elif type_slicing == '2DCor':
            if (np.unique(shape[i,:,:].ravel()).shape[0] == 2):
                count = count + 1
                positions_important_patches.append([idx, i])
        elif type_slicing == '2DSag':
            if (np.unique(shape[:,i,:].ravel()).shape[0] == 2): 
                count = count + 1
                positions_important_patches.append([idx, i])

# Create empty arrays to store the extracted patches for images and masks
vec_mask = np.zeros(shape=[count, mask_total.shape[1], mask_total.shape[2]]) 
vec_img  = np.zeros(shape=[count, mask_total.shape[1], mask_total.shape[2]]) 
print(vec_img.shape)

# Extract patches from the total dataset based on criteria
count = 0
for shape_img, shape_mask in zip(img_total, mask_total):
    for i in range(64):
        if type_slicing == '2DAxi':
            if (np.unique(shape_mask[:,:,i].ravel()).shape[0] == 2): 
                vec_mask[count] = shape_mask[:,:,i]
                vec_img[count] = shape_img[:,:,i]
                count = count + 1
                print(count, np.unique(shape_mask.ravel()))
        elif type_slicing == '2DCor':
            if (np.unique(shape_mask[i,:,:].ravel()).shape[0] == 2): 
                vec_mask[count] = shape_mask[i,:,:]
                vec_img[count] = shape_img[i,:,:]
                count = count + 1
                print(count, np.unique(shape_mask.ravel()))
        elif type_slicing == '2DSag':
            if (np.unique(shape_mask[:,i,:].ravel()).shape[0] == 2): 
                vec_mask[count] = shape_mask[:,i,:]
                vec_img[count] = shape_img[:,i,:]
                count = count + 1
                print(count, np.unique(shape_mask.ravel()))
                                                            
# Print the shapes of the extracted patch arrays
print(vec_mask.shape)
print(vec_img.shape)

# Begin loading the pre-trained segmentation model for 2D data
print('3) load model 2D')
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5]))
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = (dice_loss + (1 * focal_loss))
metrics = [sm.metrics.IOUScore(), sm.metrics.FScore()]  # Define evaluation metrics

# Get the shape of the extracted image patches
shapeX, shapeY, shapeZ = vec_img.shape

# Create an empty array to hold prediction results
y_pred = np.zeros([shapeX, shapeY, shapeZ, 1], dtype='float32')

# Use K-Fold cross-validation to predict on the extracted patches
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(vec_img)):
    print('fold ' + str(i+1))
    model = load_model(
        type_technique + '/fold' + str(i+1) + '/' + architecture + '_' + type_slicing,
        custom_objects={'iou_score': sm.metrics.IOUScore(), 'f1-score': sm.metrics.FScore(),
                        'dice_loss_plus_1binary_focal_loss': (dice_loss + (1 * focal_loss))})
    y_pred_partial = model.predict(vec_img[test_index, :, :])
    y_pred[test_index, :, :, :] = y_pred_partial

# Begin reconstructing the full predicted mask from individual patch predictions
print('unpatchify the data to single structures')
print(img_total.shape)
print(type(mask_total[0,0,0]))

# Create an empty array for the predicted mask patches
predicted_mask_patch = np.zeros(mask_total.shape, dtype='float32')
print('predicted_mask_patch')
print(predicted_mask_patch.shape)

# Reassemble patch predictions into the original mask shape
count = 0
for id_patch in positions_important_patches:
    if type_slicing == '2DAxi':
        predicted_mask_patch[id_patch[0], :, :, id_patch[1]] = y_pred[count, :, :, 0]  # >0.5 threshold could be applied here
    elif type_slicing == '2DCor':
        predicted_mask_patch[id_patch[0], id_patch[1], :, :] = y_pred[count, :, :, 0]
    elif type_slicing == '2DSag':
        predicted_mask_patch[id_patch[0], :, id_patch[1], :] = y_pred[count, :, :, 0]
    count = count + 1

# Prepare to save the reconstructed masks and original images
#print(img_full.shape)  # Debug print for full image shape, commented out
mask_predicted_full = list()
mask_full = list()
img_full = list()
size_of_all_patches = predicted_mask_patch.shape[0]
print(size_of_all_patches)
path_to_save = type_technique + '/' + type_slicing + '/pred_' + architecture + '/'
idxx = 0

# Process patches in groups to reassemble full volumes using unpatchify
for idx in range(0, size_of_all_patches, 16):
    print(idx)
    # Process predicted mask patches
    cropped_version = predicted_mask_patch[idx:idx+16, :, :, :]
    cropped_version = np.reshape(cropped_version, (4, 4, 1, 64, 64, 64))
    print(cropped_version.shape)
    mask_predicted_full.append(unpatchify(cropped_version, [256, 256, 64]))
    
    # Process original mask patches
    cropped_version_m = mask_total[idx:idx+16, :, :, :]
    cropped_version_m = np.reshape(cropped_version_m, (4, 4, 1, 64, 64, 64))
    print(cropped_version_m.shape)
    mask_full.append(unpatchify(cropped_version_m, [256, 256, 64]))
    
    # Process original image patches
    cropped_version_i = img_total[idx:idx+16, :, :, :]
    cropped_version_i = np.reshape(cropped_version_i, (4, 4, 1, 64, 64, 64))
    print(cropped_version_i.shape)
    img_full.append(unpatchify(cropped_version_i, [256, 256, 64]))
    
    # Save the predicted and true masks for the current volume
    path_tmask = path_to_save + files[idxx].split('/')[0] + '/true_mask/' + files[idxx].split('/')[1] + '.npy'
    path_pmask = path_to_save + files[idxx].split('/')[0] + '/pred_mask/' + files[idxx].split('/')[1] + '.npy'
    np.save(path_tmask, cropped_version_m)
    np.save(path_pmask, cropped_version)
    idxx = idxx + 1

# Convert lists of full volumes to numpy arrays
print('6) generate array of mask predicted')
mask_predicted_full = np.array(mask_predicted_full)
mask_full = np.array(mask_full)
img_full = np.array(img_full)

# Save the full volumes as NIfTI images
print('7) save all the images')
shape = np.load('shapes_FUSFACC.npy')
path_to_save = '/work/frayne_lab/vil/kaue/3DUnet_seg_WMH/experiment_AIinMedicine_Attention_2023/' + type_technique + '/' + type_slicing + '/pred_' + architecture + '/'
for idx, file in enumerate(files):
    print(idx, file)
    
    # Adjust volume dimensions and pad if necessary based on file type
    if file.split('/')[0] == 'Amsterdam_GE3T':
        img_full_i = img_full[idx, :shape[idx, 0], :shape[idx, 1], :]
        img_full_i = np.pad(img_full_i, ((0, 0), (0, 0), (9, 10)), 'constant')
        mask_pred = mask_predicted_full[idx, :shape[idx, 0], :shape[idx, 1], :]
        mask_pred = np.pad(mask_pred, ((0, 0), (0, 0), (9, 10)), 'constant')
        mask_full_i = mask_full[idx, :shape[idx, 0], :shape[idx, 1], :]
        mask_full_i = np.pad(mask_full_i, ((0, 0), (0, 0), (9, 10)), 'constant')
    else:
        img_full_i = img_full[idx, :shape[idx, 0], :shape[idx, 1], :shape[idx, 2]]
        mask_full_i = mask_full[idx, :shape[idx, 0], :shape[idx, 1], :shape[idx, 2]]
        mask_pred = mask_predicted_full[idx, :shape[idx, 0], :shape[idx, 1], :shape[idx, 2]]
    
    # Create NIfTI images for original image, true mask, and predicted mask
    path_file = path_to_save + file.split('/')[0] + '/orig_img/' + file.split('/')[1] + '.nii'
    path_tmask = path_to_save + file.split('/')[0] + '/true_mask/' + file.split('/')[1] + '.nii'
    path_pmask = path_to_save + file.split('/')[0] + '/pred_mask/' + file.split('/')[1] + '.nii'
    
    file_Nifti  = nib.Nifti1Image(img_full_i, affine=affines[idx])
    tmask_Nifti = nib.Nifti1Image(mask_full_i.astype('uint8'), affine=affines[idx])
    pmask_Nifti = nib.Nifti1Image(mask_pred.astype('float32'), affine=affines[idx])
    
    # Save the NIfTI images to disk
    nib.save(file_Nifti, path_file)
    nib.save(tmask_Nifti, path_tmask)
    nib.save(pmask_Nifti, path_pmask)
