import argparse
from configparser import ConfigParser
import pandas as pd
import os
import cv2
import numpy as np
from utils import normalize, generate_patches, imageNet_preprocessing
from skimage.transform import resize
from models import Resnet18
from sklearn.metrics import confusion_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True,
    help="Input file path")

    args = vars(ap.parse_args())

    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    file = args["file"]
    # base config
    base_dir = cp["BASE"].get("base_dir")
    
    # Seg config
    base_model_name = cp["CLASS"].get("model_name")
    output_weights_name = cp["CLASS"].get("output_weights_name")
    image_dimension = cp["CLASS"].getint("image_dimension")
    
    class_names = cp["CLASS"].get("class_names").split(",") 
    mask_folder = cp["CLASS"].get("mask_folder")  
    patch_size = cp["CLASS"].getint("patch_size")

    output_dir = cp["TEST"].get("output_dir")
    batch_size = cp["TEST"].getint("batch_size")
    N = cp["TEST"].getint("N")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    if use_best_weights:
        model_weights_file = os.path.join(output_dir, f"best_{output_weights_name}")    
    else:
        model_weights_file = os.path.join(output_dir, output_weights_name)

    model = Resnet18(
            input_shape=(N, patch_size, patch_size, 3),
            N = N,
            weights_path=model_weights_file,
            nb_classes=len(class_names))
    
    df = pd.read_csv(file).sample(frac=1.0)

    Y = np.zeros((len(df), ))
    Y_hat = np.zeros((len(df), ))

    for i, [filename, src, y] in enumerate(zip(df['filename'].values, df['image_folder'].values,  df[class_names].astype('int32').values)):
        img = cv2.imread(os.path.join(base_dir, "classification images", filename))
        img = normalize(img, (image_dimension, image_dimension))

        mask = cv2.imread(os.path.join(base_dir, mask_folder, filename), 0)
        mask = resize(mask, (image_dimension, image_dimension))

        img = img *  np.expand_dims(mask, -1)

        patches = generate_patches(img, mask, patch_size=(patch_size, patch_size), max_patches=N)
        patches = np.array([imageNet_preprocessing(x) for x in patches])
        x = np.expand_dims(patches, 0)

        y_hat = model.predict(x)
        
        Y[i] = np.argmax(y)
        Y_hat[i] = np.argmax(y_hat)
        print(y, y_hat)

    print(confusion_matrix(Y, Y_hat))

if __name__ == "__main__":
    main()