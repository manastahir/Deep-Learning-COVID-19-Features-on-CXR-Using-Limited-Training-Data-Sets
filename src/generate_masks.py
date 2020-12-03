import argparse
from configparser import ConfigParser
import pandas as pd
import os
import cv2
import numpy as np
from utils import normalize
from models import Densenet103
from keras.models import load_model

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
    output_dir = cp["SEG"].get("output_dir")
    base_model_name = cp["SEG"].get("model_name")
    output_weights_name = cp["SEG"].get("output_weights_name")
    image_dimension = cp["SEG"].getint("image_dimension")
    num_classes = cp["SEG"].getint("num_classes")
    mask_folder = cp["CLASS"].get("mask_folder") 

    model_weights_file = os.path.join(output_dir, f"best_{output_weights_name}")

    model = Densenet103(
            nb_classes = num_classes - 1,
            weights_path=model_weights_file,
            input_shape=(image_dimension, image_dimension, 1))
    
    model.compile(optimizer="adam", loss="binary_crossentropy")


    df = pd.read_csv(file)

    for filename, src in zip(df['filename'].values, df['image_folder'].values):
        img = cv2.imread(os.path.join(base_dir, src, filename), 0)
        img = normalize(img, (image_dimension, image_dimension), grey=True)
        img = np.expand_dims(img, axis=[0,-1])
        mask = model.predict(img)[0]*255.0
        print(filename)
        cv2.imwrite(os.path.join(base_dir, mask_folder, filename), mask)


if __name__ == "__main__":
    main()