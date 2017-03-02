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

import util
reload(util)


def predict(image, model):
    ans = np.zeros_like(image)
    for i in range(5):
        for j in range(5):
            row_slice = slice(i * 96, (i + 1) * 96)
            col_slice = slice(j * 96, (j + 1) * 96)
            image_patch = np.reshape(image[row_slice, col_slice], [1, 1, 96, 96])
            image_patch = luna_train_unet2.normalize_images(image_patch)
            mask_patch = model.predict(image_patch)[0,0]
            ans[row_slice, col_slice] = mask_patch
    return ans

def pred_nodule_mask(image, model):
    ans = np.zeros_like(image).astype(np.float16)
    nrows, ncols = np.ceil((np.asarray(image.shape) - 96) / 24.0).astype(np.int)
    for i in range(nrows):
        for j in range(ncols):
            row_slice = slice(24 * i, 24 * i + 96)
            col_slice = slice(24 * j, 24 * j + 96)
            image_patch = np.reshape(image[row_slice, col_slice], [1, 1, 96, 96])
            image_patch = luna_train_unet2.normalize_images(image_patch)
            mask_patch = model.predict(image_patch)[0,0]
            ans[row_slice, col_slice] += mask_patch
    return np.minimum(1.0, ans / 4.0)


def segment(p_images, module_model, spacing=1):
    #p_lungs = util.segment_lung_mask_v2(p_images, spacing)
    module_mask = np.zeros_like(p_images)
    for slice in range(len(p_images)):
        slice = 10
        img = p_images[slice]
        module_mask[slice] = pred_nodule_mask(img, module_model)

        break

    return module_mask

