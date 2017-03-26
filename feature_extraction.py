# -*- coding: utf-8 -*-
from skimage import measure

# script to process

import os

import csv
import numpy as np
import scipy as sp

from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb
from matplotlib import pyplot as plt

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

FEATURES = ['area', 'convex_area', 'eccentricity', 'equivalent_diameter',
    'extent', 'major_axis_length', 'minor_axis_length', 'orientation',
    'perimeter', 'value_mean', 'value_std']

NODULE_SIZE_THRESHOLD = 5

#_MODEL = None
#with tf.device('/gpu:0'):
    #_MODEL = luna_train_unet2.get_unet()
#_MODEL.load_weights('./unet2.hdf5')

patient_names = os.listdir(_DATA_DIR)
patient_labels = read_patient_labels(_LABELS_CSV)
test_patient_names = list(set(read_patient_labels(_SAMPLE_CSV).keys()))

print("[ground-truth patients, test-patients] = [%d, %d]"
    % (len(patient_labels), len(test_patient_names)))


def get_patient_feature_data(patients):
    feature_dict = {}
    for pname in patients:
        outfile = os.path.join(
            _SEGMENTATION_DIR, pname + '.nodule_cords')

        if not os.path.exists(outfile):
            print 'output file not exist: {}'.format(outfile)
            continue

        # get patient image dir
        #print 'processing patient: %s' % pname
        patient_img_dir = os.path.join(_DATA_DIR, pname)
        #print 'patient_img_dir: %s' % patient_img_dir
        p_images = lung_image.load_image_from_patient_dir(patient_img_dir)
        #plt.figure('loaded image')
        #plt.imshow(p_images[10], cmap='gray')
        #plt.show()

        module_cords = np.load(outfile)

        features = extract_features(p_images, module_cords)
        feature_dict[pname] = features
    return feature_dict


def extract_features(p_images, module_cords):
    #print p_images.shape

    # crete module mask image
    module_mask = np.zeros_like(p_images)
    #print module_cords[6]
    #print module_cords

    for s in range(len(p_images)):
        cords = module_cords[s]
        #print 'cords[0]'
        #print cords[0]
        if not cords[0].any():
            continue

        #print s
        #print cords[0]
        #print cords[1]

        module_mask[s][cords[0], cords[1]] = 1
        # verify the segmentation result
        #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        #ax1.imshow(p_images[s], cmap='gray')
        #ax1.set_title('image')
        #ax2.imshow(module_mask[s], cmap='gray')
        #ax1.set_title('segmentation, pixels: {}'.format(len(cords[0])))
        #plt.show()

    mask_labels = measure.label(module_mask, background=0)

    # for each slice, get the features,
    # use the label to group results from different slices
    label_feature_maps = {}
    for s in range(len(p_images)):
        p_image = p_images[s]
        mask_label_slice = mask_labels[s]
        properties = measure.regionprops(mask_label_slice)
        if not properties:
            continue

        #print 'found valid properties'
        for prop in properties:
            label_feature_map = label_feature_maps.get(prop.label, {})
            values = p_image[prop.coords[:, 0], prop.coords[:, 1]]

            for field in FEATURES:
                feature_list = label_feature_map.get(field, [])
                f = None
                if field == 'value_mean':
                    f = np.mean(values)
                elif field == 'value_std':
                    f = np.std(values)
                else:
                    f = getattr(prop, field)
                feature_list.append(f)
                label_feature_map[field] = feature_list

            label_feature_maps[prop.label] = label_feature_map

    sizes = {label: np.sum(feature['area'])
        for label, feature in label_feature_maps.items()}
    #print '[all nodules = {}, large nodules = {}]'.format(
        #len(sizes), sum(s >= NODULE_SIZE_THRESHOLD for s in sizes.values()))
    label_feature_maps = {
        label: feature for label, feature in label_feature_maps.items()
        if sizes[label] >= NODULE_SIZE_THRESHOLD}

    features_all_labels = []
    for label, feature in label_feature_maps.items():
        label_feature_summary = {
            'area': np.sum(feature['area']),
            'equivalent_diameter': np.sum(feature['equivalent_diameter']),
            'convex_area': np.sum(feature['convex_area']),
            'area': np.sum(feature['area']),
            'perimeter': np.dot(feature['perimeter'], feature['area']),
            'extent': np.dot(feature['extent'], feature['area']),
            'major_axis_length': np.dot(
                feature['major_axis_length'], feature['area']),
            'minor_axis_length': np.dot(
                feature['minor_axis_length'], feature['area']),
            'eccentricity': np.dot(feature['eccentricity'], feature['area']),
            'orientation': np.dot(feature['orientation'], feature['area']),
            'value_mean': np.dot(feature['value_mean'], feature['value_mean']),
            'value_std': np.dot(feature['value_std'], feature['value_std'])}
        features_all_labels.append(label_feature_summary)

    ## features
    features = []
    for field in FEATURES:
        values = [label_feature[field] for label_feature in features_all_labels]
        if not values:
            values = [0, 0]
        features.extend(
            [np.max(values),
             np.min(values),
             np.sum(values),
             np.mean(values),
             np.std(values)])

    print 'number of nodules: {}'.format(len(label_feature_maps.keys()))
    #print features
    return features

if __name__ == '__main__':
    patients_tr = patient_labels.keys()
    patients_te = test_patient_names

    patients = set(patients_tr) | set(patients_te)
    print("[ground-truth patients, test-patients, all] = [%d, %d, %d]"
     % (len(patients_tr), len(patients_te), len(patients)))

    #patients = ['aec5a58fea38b77b964007aa6975c049']
    feature_map = get_patient_feature_data(patients)
    with file('../../data/feature_map.dat', 'w') as outfile:
            np.save(outfile, feature_map)