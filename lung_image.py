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

import feature_extraction

import util
reload(util);

def load_image_from_patient_dir(patient_image_dir):
    return 'this is a image from dir %s' % patient_image_dir
    
    p_slices = _load_scans(patient_image_dir)
    p_images = _get_pixels_hu(p_slices)
    p_images_resampled = util.resample(p_images, _get_spacing(p_slices[0]), [1, 1, 1])
    return p_images_resampled


def save(**kwargs):
    with open('save.dat', 'wb') as f:
        pickle.dump(kwargs, f)

def load():
    with open('save.dat', 'rb') as f:
        return pickle.load(f)



def _load_scans(patient_image_dir):
    slices = [dicom.read_file(os.path.join(patient_image_dir, s)) for s in os.listdir(patient_image_dir)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                                 slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation -
                                 slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def _get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    outside_idx = (image == -2000)
    
    # Convert to Hounsfield units (HU)
    intercept = float(scans[0].RescaleIntercept)
    slope = float(scans[0].RescaleSlope)
    
    image = (slope * image + intercept).astype(np.int16)
    image[outside_idx] = -1000  # HU of air
    
    return image

def _get_spacing(scan):
    # z, y, x
    ans = [scan.SliceThickness, scan.PixelSpacing[1], scan.PixelSpacing[0]]
    return [float(s) for s in ans]