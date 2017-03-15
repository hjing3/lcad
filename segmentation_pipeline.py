# script to process

import os

import csv
import numpy as np
import scipy as sp

#from sklearn.cross_validation import StratifiedKFold as KFold
#from sklearn.metrics import classification_report
#from sklearn.ensemble import RandomForestClassifier as RF


import feature_extraction
import lung_image
import luna_preprocess as prep
import luna_train_unet2
import util
import segmentation
import tensorflow as tf

import os
import sys
import csv
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import morphology
from glob import glob


def read_patient_labels(csv_fname):
    with open(csv_fname) as f:
        reader = csv.reader(f, delimiter=',')
        reader.next()  # skip header
        return dict([(r[0], float(r[1])) for r in reader if len(r) == 2])


_DATA_DIR = '../../data/stage1'
_SEGMENTATION_DIR = '../../data/stage1_preprocess/unet2_segmentation_2'
_OUTPUT_DIR = '../../data/stage1_preprocess'
_LABELS_CSV = '../../data/stage1_labels.csv'
_SAMPLE_CSV = '../../data/stage1_sample_submission.csv'

_MODEL = None
with tf.device('/gpu:0'):
    _MODEL = luna_train_unet2.get_unet()
_MODEL.load_weights('./unet2.hdf5')

patient_names = os.listdir(_DATA_DIR)
patient_labels = read_patient_labels(_LABELS_CSV)
test_patient_names = list(set(read_patient_labels(_SAMPLE_CSV).keys()))

print("[ground-truth patients, test-patients] = [%d, %d]" \
    % (len(patient_labels), len(test_patient_names)))


def get_nodule_segmentation():
    patients = patient_labels.keys()
    patients.extend(test_patient_names)

    print patients

    for pname in patients:
        outfile = os.path.join(
            _SEGMENTATION_DIR, pname + '.nodule_cords')
        if os.path.exists(outfile):
            continue

        print('segmentation for patient: {}'.format(pname))
        patient_img_dir = os.path.join(_DATA_DIR, pname)
        p_images = lung_image.load_image_from_patient_dir(patient_img_dir)

        # is it ok to assume spacing to be 1.0 after the resampling???
        # can use preprocess module as a reference
        spacing = 1.0
        module_cords = segmentation.segment(p_images, _MODEL, spacing)

        outfile_name = patient_img_dir = os.path.join(
            _SEGMENTATION_DIR, pname + '.nodule_cords')
        with file(outfile_name, 'w') as outfile:
            np.save(outfile, module_cords)

        print('segmentation done, result saved to: {}'.format(outfile))

if __name__ == '__main__':
    get_nodule_segmentation()
