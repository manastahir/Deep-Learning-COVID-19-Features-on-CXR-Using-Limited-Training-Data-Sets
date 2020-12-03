import time
import numpy as np
import pandas as pd
import cv2
from skimage.transform import resize
from sklearn.feature_extraction.image import extract_patches_2d
import time

def get_class_weights(total_counts, class_positive_counts, multiply):
    """
    Calculate class_weight used in training
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights

def get_class_counts(datafile, class_names=None):
  data = pd.read_csv(datafile)
  T  = data.shape[0]
  P = 0
  if(class_names is not None):
    labels = data[class_names].values
    P = np.sum(labels, axis=0)
    P = dict(zip(class_names, P))

  return T, P

def normalize(img, target_size, grey=False):
    img = resize(img, target_size)
    img = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    #Histogram equaliztion of the given image
    if(grey):
        img = cv2.equalizeHist(img) 
    else:
        for i in range(img.shape[-1]):
            img[:,:,i] = cv2.equalizeHist(img[:,:,i]) 
    #Gamma correction with gamma = 0.5
    gamma = 0.5
    table = np.array([(np.float_power(i/255.0, gamma)) * 255 for i in np.arange(0, 256)]).astype("float32")
    img = cv2.LUT(img, table)

    img = img/255.0

    return img

def generate_patches(img, mask, patch_size = (224, 224), max_patches=100):
    half_patch_size = (int(patch_size[0]/2), int(patch_size[1]/2))
    upper_limit = (img.shape[0] - half_patch_size[0], img.shape[1] - half_patch_size[1])
    lower_limit = half_patch_size

    #select a random 224x224 patch
    patch_array = []
    for i in range(max_patches):    
        while(True):
            x, y = np.random.randint(lower_limit[0], upper_limit[0]), np.random.randint(lower_limit[1], upper_limit[1])
            if(mask[x, y] != 0):
                patch = img[x-half_patch_size[0]:x+half_patch_size[0], y-half_patch_size[1]:y+half_patch_size[1]]
                break
        patch_array.append(patch)
    
    patch_array = np.array(patch_array)

    return patch_array

def imageNet_preprocessing(img):
    if(np.max(img) > 1.0):
        img = img/255.0
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    img = (img - means)/ stds
    
    return img