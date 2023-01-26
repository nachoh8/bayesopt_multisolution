import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def levels_from_image(filename):
    img=mpimg.imread(filename)
    img=rgb2gray(img)

    levels = {}
    error = 1e-5
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            key = 1.0-img[i,j] # Darker == Higher cost

            # Just in case of floating point errors
            if len(levels.keys()) > 0:
                closest = min(levels.keys(), key=lambda x:abs(x-key))
                if abs(key - closest) < error:
                    key = closest

            if key not in levels:
                levels[key] = []
            levels[key].append([j,img.shape[0]-i-1]) # Coordinates starting in top left


    sorted_keys = sorted(levels.keys())
    sorted_levels = []
    for i in range(len(sorted_keys)):
        sorted_levels.append(levels[sorted_keys[i]])

    return sorted_levels, sorted_keys, img.shape

def masks_from_image(filename):
    img=mpimg.imread(filename)
    img=rgb2gray(img)

    masks = {}
    error = 1e-5
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            key = 1.0-img[i,j] # Darker == Higher cost

            # Just in case of floating point errors
            if len(masks.keys()) > 0:
                closest = min(masks.keys(), key=lambda x:abs(x-key))
                if abs(key - closest) < error:
                    key = closest

            if key not in masks:
                masks[key] = np.zeros(img.shape)
            masks[key][j,img.shape[0]-i-1] = 1.0  # Coordinates starting in top left


    sorted_keys = sorted(masks.keys())
    sorted_masks = []
    for i in range(len(sorted_keys)):
        sorted_masks.append(masks[sorted_keys[i]])

    return sorted_masks, sorted_keys, img.shape

