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
import kagl_preprocess
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
_SEGMENTATION_DIR = '/mnt/sdb1/backup/kaggle/lungcancer/stage1/'
_OUTPUT_DIR = '../../data/stage1_preprocess'
_LABELS_CSV = '../../data/stage1_labels.csv'
_SAMPLE_CSV = '../../data/stage1_sample_submission.csv'

FEATURES = ['area', 'convex_area', 'eccentricity', 'equivalent_diameter',
    'extent', 'major_axis_length', 'minor_axis_length', 'orientation',
    'perimeter', 'value_mean', 'value_std']

NODULE_SIZE_THRESHOLD = 7

patient_names = os.listdir(_DATA_DIR)
patient_labels = read_patient_labels(_LABELS_CSV)
test_patient_names = list(set(read_patient_labels(_SAMPLE_CSV).keys()))

print("[ground-truth patients, test-patients] = [%d, %d]"
    % (len(patient_labels), len(test_patient_names)))

IMAGE_UTIL = kagl_preprocess.Image('stage1')


def load_image_and_masks(patient_name):
    ans = np.load(os.path.join(
        _SEGMENTATION_DIR, 'stage1_test_unet5', patient_name + '_mask.npz'))
    nodule_mask = ans['nodule_mask']

    IMAGE_UTIL.load(
        patient_name, '/mnt/sdb1/backup/kaggle/lungcancer/stage1/stage1_output')
    return IMAGE_UTIL.masked_lung, IMAGE_UTIL._lung_mask, nodule_mask


def get_patient_feature_data(patients):
    feature_dict = {}
    for pname in patients:
        try:
            images, lung_masks, nodule_masks = load_image_and_masks(pname)
            nodule_masks[lung_masks < 0.5] = 0
            nodule_masks[nodule_masks < 0.5] = 0
            nodule_masks[nodule_masks > .1] = 1
        except:
            print 'corrupted file: {}'.format(pname)

        features = extract_features(images, lung_masks, nodule_masks)
        feature_dict[pname] = features
    return feature_dict


def extract_features(p_images, lung_masks, nodule_masks):
    mask_labels = measure.label(nodule_masks, background=0)

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
            'value_mean': np.dot(feature['value_mean'], feature['area']),
            'value_std': np.dot(feature['value_std'], feature['area'])}
        features_all_labels.append(label_feature_summary)

    ## features
    features = [len(label_feature_maps.keys())]
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

    # calculate the dilated nodule area in whole image (all slices)
    nodule_mask_dilated = morphology.binary_dilation(
        nodule_masks, np.ones((5, 5, 5)))
    values_dilated = p_images[nodule_mask_dilated > 0]
    values_nodule = p_images[nodule_masks > 0]
    values_bkg = p_images[nodule_mask_dilated > nodule_masks]

    contrast = np.mean(values_nodule) - np.mean(values_bkg)
    nodule_mean = np.mean(values_nodule)
    nodule_std = np.std(values_nodule)
    bkg_mean = np.mean(values_bkg)
    bkg_std = np.std(values_bkg)
    area_mean = np.mean(values_dilated)
    area_std = np.std(values_dilated)
    new_features = [nodule_mean, nodule_std, bkg_mean, bkg_std,
        area_mean, area_std, contrast]
    #print new_features
    features.extend(new_features)

    # calculate gradient in the dilated area, orientation and magnitude
    nodule_area_coord = np.where(nodule_mask_dilated > .5)
    x_min = np.min(nodule_area_coord[0])
    x_max = np.max(nodule_area_coord[0])
    y_min = np.min(nodule_area_coord[1])
    y_max = np.max(nodule_area_coord[1])
    z_min = np.min(nodule_area_coord[2])
    z_max = np.max(nodule_area_coord[2])
    img_nodule_area = p_images[x_min:x_max, y_min:y_max, z_min:z_max]
    mask_dilated_nodule_area = nodule_mask_dilated[
        x_min:x_max, y_min:y_max, z_min:z_max]
    mask_nodule_area = nodule_masks[x_min:x_max, y_min:y_max, z_min:z_max]
    gradient_nodule_area = np.gradient(img_nodule_area)

    location_area = mask_dilated_nodule_area > 0
    gradient_area_x = gradient_nodule_area[0][location_area]
    gradient_area_y = gradient_nodule_area[1][location_area]
    gradient_area_z = gradient_nodule_area[2][location_area]
    new_features = [
        np.mean(gradient_area_x), np.std(gradient_area_x),
        np.mean(np.abs(gradient_area_x)), np.std(np.abs(gradient_area_x)),
        np.mean(gradient_area_y), np.std(gradient_area_y),
        np.mean(np.abs(gradient_area_y)), np.std(np.abs(gradient_area_y)),
        np.mean(gradient_area_z), np.std(gradient_area_z),
        np.mean(np.abs(gradient_area_z)), np.std(np.abs(gradient_area_z)),
        ]
    features.extend(new_features)
    #print new_features

    # distance from lung wall, and location (one-hot, 1 of six directions)
    properties = measure.regionprops(mask_labels)
    location_features_dict = {}
    for prop in properties:
        location_features = get_location_feature(lung_masks, prop)
        if not location_features:
            location_features = [0] * 8
        for lf_ind in range(len(location_features)):
            old_feature = []
            if lf_ind in location_features_dict:
                old_feature = location_features_dict[lf_ind]
            old_feature.append(location_features[lf_ind])
            location_features_dict[lf_ind] = old_feature

    #print 'location features (location_features_dict):'
    #print location_features_dict

    location_features = []
    for lf_ind in range(len(location_features_dict.keys())):
        values = location_features_dict[lf_ind]
        location_features.extend(
            [np.max(values),
             np.min(values),
             np.sum(values),
             np.mean(values),
             np.std(values)])

    #print 'location_features:'
    #print location_features
    features.extend(location_features)

    #print 'number of nodules: {}'.format(len(label_feature_maps.keys()))
    #print features
    return features


def get_location_feature(lung_mask, prop):
    orientations = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    x = int(np.mean(prop.coords[:, 0]))
    y = int(np.mean(prop.coords[:, 1]))
    z = int(np.mean(prop.coords[:, 2]))

    closest_orientation_index = 0
    dist_small = -1
    close_orientation = -1
    dist_large = -1
    for index in range(len(orientations)):
        orientation = orientations[index]
        dist = 0
        while lung_mask[
            x + dist * orientation[0],
            y + dist * orientation[1],
            z + dist * orientation[2]] > .5:
            dist += 1
        if dist < dist_small or index == 0:
            dist_small = dist
            close_orientation = index
        if dist > dist_large or index == 0:
            dist_large = dist

    closest_orientations = [0.0] * 6
    closest_orientations[close_orientation] = 1.0
    locatoin_features = [
        dist_small * 1.0, dist_large * 1.0] + closest_orientations
    #print 'locaton features: '
    #print locatoin_features
    #return locatoin_features


if __name__ == '__main__':
    patients_tr = patient_labels.keys()
    patients_te = test_patient_names

    patients = set(patients_tr) | set(patients_te)
    print("[ground-truth patients, test-patients, all] = [%d, %d, %d]"
     % (len(patients_tr), len(patients_te), len(patients)))

    #patients = ['aec5a58fea38b77b964007aa6975c049']
    feature_map = get_patient_feature_data(patients)

    with file(
        '../../data/feature_map_thres{}.dat'.format(
            NODULE_SIZE_THRESHOLD), 'w') as outfile:
            np.save(outfile, feature_map)