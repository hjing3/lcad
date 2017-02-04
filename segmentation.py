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

def segment_nodule(p_lungs):
    return p_lungs

def segment(p_images):
    return p_images

    p_lungs = util.segment_lung_mask_v2(p_images, 1)
    p_nodule = segment_nodule(p_lungs)

