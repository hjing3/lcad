# script to process

import os

import csv
import numpy as np
import scipy as sp

from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb

import feature_extraction
import lung_image
import segmentation


def read_patient_labels(csv_fname):
    with open(csv_fname) as f:
        reader = csv.reader(f, delimiter=',')
        reader.next()  # skip header
        return dict([(r[0], float(r[1])) for r in reader if len(r) == 2])


_DATA_DIR = '/home/haojing/projects/kaggle/lungcancer/data/stage1'
_LABELS_CSV = '/home/haojing/projects/kaggle/lungcancer/data/stage1_labels.csv'
_SAMPLE_CSV = \
    '/home/haojing/projects/kaggle/lungcancer/data/stage1_sample_submission.csv'

patient_names = os.listdir(_DATA_DIR)
patient_labels = read_patient_labels(_LABELS_CSV)
test_patient_names = set(read_patient_labels(_SAMPLE_CSV).keys())

print "[ground-truth patients, test-patients] = [%d, %d]" \
    % (len(patient_labels), len(test_patient_names))


def get_patient_feature_data(patients):
    feature_dict = {}
    for pname in patient_labels.keys():
        # get patient image dir
        print 'processing patient: %s' % pname
        patient_img_dir = os.path.join(_DATA_DIR, pname)
        print 'patient_img_dir: %s' % patient_img_dir
        p_images = lung_image.load_image_from_patient_dir(patient_img_dir)
        print 'pimages: %s' % p_images

        p_segment = segmentation.segment(p_images)
        print 'p_segment: %s' % p_segment

        p_feature = feature_extraction.extract_features(p_segment)
        print 'p_feature: %s' % str(p_feature)
        feature_dict[pname] = p_feature
    return feature_dict


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act)
        * sp.log(sp.subtract(1, pred)))
    ll = ll * - 1.0 / len(act)
    return ll


def classify_data():
    patients = patient_labels.keys()
    X_map = get_patient_feature_data(patients)
    X = np.array([X_map[p] for p in patients])
    Y = np.array([patient_labels[p] for p in patients])

    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = \
            X[train, :], X[test, :], Y[train], Y[test]
        clf = RF(n_estimators=100, n_jobs=3)
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss", logloss(Y, y_pred))

    # All Cancer
    print "Predicting all positive"
    y_pred = np.ones(Y.shape)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss", logloss(Y, y_pred))

    # No Cancer
    print "Predicting all negative"
    y_pred = Y * 0
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss", logloss(Y, y_pred))

    # try XGBoost
    print ("XGBoost")
    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = \
            X[train, :], X[test, :], Y[train], Y[test]
        clf = xgb.XGBClassifier(objective="binary:logistic")
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    print("logloss", logloss(Y, y_pred))

if __name__ == '__main__':
    classify_data()
