# -*- coding: utf-8 -*-
from skimage import measure


def extract_features(p_images, p_masks):
    all_labels = measure.label(p_masks)
    properties = measure.regionprops(all_labels)
    sizes = [prop.area for prop in properties]

    # add more features
    return [len(sizes)]