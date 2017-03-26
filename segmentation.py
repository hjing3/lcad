# -*- coding: utf-8 -*-


# script to process

import os
import sys
import time
import ctypes
import multiprocessing
import warnings

import dicom
import csv
import pickle
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf

from copy import deepcopy
from collections import namedtuple
from scipy.misc import imread
from scipy import stats
from scipy import ndimage
from skimage import measure
from skimage import morphology
from skimage import feature
from skimage import data
from skimage.measure import label, regionprops, perimeter
from skimage.filters import roberts, sobel
from skimage.morphology import ball, disk, dilation, binary_erosion
from skimage.morphology import remove_small_objects, erosion, closing
from skimage.morphology import reconstruction, binary_closing
from skimage.morphology import binary_dilation, binary_opening
from skimage.segmentation import clear_border
from matplotlib import pyplot as plt
import luna_train_unet2

import feature_extraction

import luna_train_unet5

import util
reload(util)

WIDTH = 8
BOARD_MASK = np.ones((96, 96)).astype(np.float16)
BOARD_MASK[:, 0:8] = 0
BOARD_MASK[:, 96-8:96] = 0
BOARD_MASK[0:8, :] = 0
BOARD_MASK[96-8:96, :] = 0


def pred_nodule_mask_org(image, model):
    ans = np.zeros_like(image).astype(np.float16)
    nrows, ncols = np.ceil((np.asarray(image.shape) - 96) / 24.0).astype(np.int)
    for i in range(nrows):
        for j in range(ncols):
            row_slice = slice(24 * i, 24 * i + 96)
            col_slice = slice(24 * j, 24 * j + 96)
            image_patch = np.reshape(image[row_slice, col_slice], [1, 1, 96, 96])
            #image_patch = luna_train_unet2.normalize_images(image_patch)
            mask_patch = model.predict(image_patch)[0,0]
            ans[row_slice, col_slice] += mask_patch
    return np.minimum(1.0, ans / 4.0)


def pred_nodule_mask(image, model):
    ans = np.zeros_like(image).astype(np.float16)
    step = 80
    #??? Are these number correct when applied to kaggle data?
    nrows, ncols = np.ceil((np.asarray(image.shape) - 96) / (step*1.0)).astype(np.int)
    for i in range(nrows):
        for j in range(ncols):
            row_slice = slice(step * i, step * i + 96)
            col_slice = slice(step * j, step * j + 96)
            image_patch = np.reshape(image[row_slice, col_slice], [1, 1, 96, 96])
            #image_patch = luna_train_unet2.normalize_images(image_patch)
            mask_patch = model.predict(image_patch)[0, 0]
            mask_patch = mask_patch * BOARD_MASK
            
            ans[row_slice, col_slice] += mask_patch
    return ans




def pred_nodule_mask_batch(image, model):
    """Patching INCORRECT"""
    step = 48.0
    times = 96 / step

    ans = np.zeros_like(image).astype(np.float16)
    #??? Are these number correct when applied to kaggle data?
    nrows, ncols = np.ceil((np.asarray(image.shape) - 96) / step).astype(np.int)

    image_patches = []

    #print 'generate image patches for unet prediction'
    for i in range(nrows):
        for j in range(ncols):
            row_slice = slice(step * i, step * i + 96)
            col_slice = slice(step * j, step * j + 96)
            image_patch = np.reshape(
                image[row_slice, col_slice], [1, 1, 96, 96])
            image_patch = luna_train_unet2.normalize_images(image_patch)
            image_patches.append(image_patch)

    batch = 10
    patches = []
    mask_patches_all = []
    n_batches = 1 + len(image_patches) / batch
    for i in range(n_batches):
        #print 'processing patch batch {} of {}'.format(i, n_batches)
        patches = image_patches[
            i * batch: min(i * batch + batch, len(image_patches))]
        if not patches:
            break
        patches = np.concatenate(patches)
        mask_patches = model.predict(patches)
        mask_patches_all.extend(mask_patches)

    #print 'put prediction patches together'
    index_patch = 0
    for i in range(nrows):
        for j in range(ncols):
            row_slice = slice(step * i, step * i + 96)
            col_slice = slice(step * j, step * j + 96)
            ans[row_slice, col_slice] += mask_patches_all[index_patch][0, 0]
            index_patch += 1
    return np.minimum(1.0, ans / times)


def segment(p_images, module_model, spacing=1):
    module_slice_mask = []

    img_mean = -520.0
    img_std = 520.0
    #img_mean = np.mean(p_images)
    #img_std = np.std(p_images)
    p_images = (p_images - img_mean) / img_std
    
    for slice in range(len(p_images)):
        if slice % 10 == 0:
            print 'segment of slice {} of {}'.format(slice, len(p_images))
        img = p_images[slice]
        module_mask = pred_nodule_mask(img, module_model)
        #module_mask = luna_train_unet5.pred_nodule_mask(img, module_model)

        nodule_coords = np.where(module_mask > 0)
        module_slice_mask.append(nodule_coords)

        #print nodule_coords

        #if nodule_coords[0].any():
            #print nodule_coords
            #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            #ax1.imshow(img, cmap='gray',aspect='auto')
            #ax2.imshow(module_mask > 0, cmap='gray',aspect='auto')
            #plt.show()
    return module_slice_mask
