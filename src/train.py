import json
import shutil
import os
import pickle
import argparse
from configparser import ConfigParser
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.models import save_model, load_model
from keras.metrics import *

from data_generators import classification_gen, segmentation_gen
from models import Densenet103, Resnet18
from callback import MultiGPUModelCheckpoint, MultipleClassAUROC, Jaccard
from utils import get_class_counts, get_class_weights, imageNet_preprocessing
from augmenter import augmenter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", required=True,
	  help="Train type")
    args = vars(ap.parse_args())

    #parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    TRAIN = args["train"]
    if(TRAIN not in ["SEG", "CLASS"]):
        raise ValueError(f"{TRAIN} not defined")

    # base config
    base_dir = cp["BASE"].get("base_dir")

    # train config
    output_dir = cp[TRAIN].get("output_dir")
    base_model_name = cp[TRAIN].get("model_name")

    use_base_model_weights = cp[TRAIN].getboolean("use_base_model_weights")
    use_trained_model_weights = cp[TRAIN].getboolean("use_trained_model_weights")
    use_best_weights = cp[TRAIN].getboolean("use_best_weights")

    output_weights_name = cp[TRAIN].get("output_weights_name")

    epochs = cp[TRAIN].getint("epochs")
    batch_size = cp[TRAIN].getint("batch_size")
    initial_learning_rate = cp[TRAIN].getfloat("initial_learning_rate")

    generator_workers = cp[TRAIN].getint("generator_workers")
    image_dimension = cp[TRAIN].getint("image_dimension")

    train_steps = cp[TRAIN].get("train_steps")
    validation_steps = cp[TRAIN].get("validation_steps")

    patience_reduce_lr = cp[TRAIN].getint("patience_reduce_lr")
    patience_early_stop = cp[TRAIN].getint("patience_early_stop")
    min_lr = cp[TRAIN].getfloat("min_lr")
    
    dataset_csv_dir = cp[TRAIN].get("dataset_csv_dir")

    show_model_summary = cp[TRAIN].getboolean("show_model_summary")

    if(TRAIN == "CLASS"):
        positive_weights_multiply = cp[TRAIN].getfloat("positive_weights_multiply")
        class_names = cp[TRAIN].get("class_names").split(",") 
        mask_folder = cp[TRAIN].get("mask_folder")  
        patch_size = cp[TRAIN].getint("patch_size") 
        N = cp["TEST"].getint("N")
    else:
        num_classes = cp[TRAIN].getint("num_classes")
        class_names = None

    current_epoch = 0
    # check output_dir, create it if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # if previously trained weights is used, never re-split
    if use_trained_model_weights:
        # resuming mode
        print("** use trained model weights **")
        # load training status for resuming
        training_stats_file = os.path.join(output_dir, ".training_stats.json")

        if os.path.isfile(training_stats_file):
            training_stats = json.load(open(training_stats_file))
            initial_learning_rate = training_stats['lr']
            current_epoch = training_stats['epoch']
        else:
            training_stats = {}
    else:
        # start over
        training_stats = {}
    

    print(f"backup config file to {output_dir}")
    shutil.copy(config_file, os.path.join(output_dir, os.path.split(config_file)[1]))

    datasets = ["train", "val", "test"]
    for dataset in datasets:
        shutil.copy(os.path.join(dataset_csv_dir, f"{dataset}.csv"), output_dir)
    
    # get train/dev sample counts
    train_counts, train_pos_counts = get_class_counts(os.path.join(output_dir,"train.csv"), class_names)
    val_counts, _ = get_class_counts(os.path.join(output_dir,"val.csv"), class_names)
    
    # compute steps
    if train_steps == "auto":
        train_steps = int(train_counts / batch_size)
    else:
        try:
            train_steps = int(train_steps)
        except ValueError:
            raise ValueError(f"train_steps: {train_steps} is invalid, please use 'auto' or integer.")

    print(f"** train_steps: {train_steps} **")

    if validation_steps == "auto":
        validation_steps = int(val_counts / batch_size)
    else:
        try:
            validation_steps = int(validation_steps)
        except ValueError:
            raise ValueError(f"validation_steps: {validation_steps} is invalid,please use 'auto' or integer.")

    print(f"** validation_steps: {validation_steps} **")

    class_weights = None
    if(TRAIN == "CLASS"):
        # compute class weights
        print("** compute class weights from training data **")
        class_weights = get_class_weights(
            train_counts,
            train_pos_counts,
            multiply=positive_weights_multiply,
        )

    print("** class_weights **")
    print(class_weights)

    
    print("** load model **")

    if use_trained_model_weights:
        if use_best_weights:
            model_weights_file = os.path.join(output_dir, f"best_{output_weights_name}")    
        else:
            model_weights_file = os.path.join(output_dir, output_weights_name)
    else:
        model_weights_file = None
    

    print("** compile model **")
    METRICS = [
            TruePositives(name='tp'),
            FalsePositives(name='fp'),
            TrueNegatives(name='tn'),
            FalseNegatives(name='fn'), 
            Accuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
    ]
    optimizer = Adam(lr=initial_learning_rate)

    if(TRAIN == "CLASS"):
        model = Resnet18(
            input_shape=(N, patch_size, patch_size, 3),
            weights_path=model_weights_file,
            N = N,
            nb_classes=len(class_names))
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=METRICS)
        checkpoint_monitor = 'val_loss'
    else:
        model = Densenet103(
            nb_classes = num_classes-1,
            weights_path=model_weights_file,
            input_shape=(image_dimension, image_dimension, 1))
        model.compile(optimizer=optimizer, loss="binary_crossentropy")
        checkpoint_monitor = 'val_loss'

    if show_model_summary:
        print(model.summary())
    
    print("** create image generators **")

    if(TRAIN == "CLASS"):
        train_sequence = classification_gen(
            dataset_csv_file=os.path.join(output_dir, "train.csv"),
            class_names=class_names,
            N = N,
            batch_size=batch_size,
            normalization_func = imageNet_preprocessing,
            target_size=(image_dimension, image_dimension),
            patch_size=(patch_size, patch_size),
            augmenter=augmenter,
            base_dir=base_dir,
            mask_folder=mask_folder,
            steps=train_steps,
        )

        validation_sequence = classification_gen(
            dataset_csv_file=os.path.join(output_dir, "val.csv"),
            class_names=class_names,
            N = N,
            batch_size=batch_size,
            normalization_func = imageNet_preprocessing, 
            target_size=(image_dimension, image_dimension),
            patch_size=(patch_size, patch_size),
            augmenter=augmenter,
            base_dir=base_dir,
            steps=validation_steps,
            mask_folder=mask_folder,
            shuffle_on_epoch_end=False,
        )
    else:
        train_sequence = segmentation_gen(
            dataset_csv_file=os.path.join(output_dir, "train.csv"),
            batch_size=batch_size,
            target_size=(image_dimension, image_dimension),
            augmenter=augmenter,
            base_dir=base_dir,
            steps=train_steps,
        )

        validation_sequence = segmentation_gen(
            dataset_csv_file=os.path.join(output_dir, "val.csv"),
            batch_size=batch_size,
            target_size=(image_dimension, image_dimension),
            augmenter=augmenter,
            base_dir=base_dir,
            steps=validation_steps,
            shuffle_on_epoch_end=False,
        )

    output_weights_path = os.path.join(output_dir, output_weights_name)
    print(f"** set output weights path to: {output_weights_path} **")

    checkpoint = ModelCheckpoint(
        output_weights_path,
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
        monitor=checkpoint_monitor
        )

    if(TRAIN == "CLASS"):
        performance_callback = MultipleClassAUROC(
            sequence=validation_sequence, 
            class_names=class_names, 
            weights_path=output_weights_path,
            stats=training_stats,
            workers=generator_workers
            )
    else:
        performance_callback = Jaccard(
            sequence=validation_sequence,
            weights_path=output_weights_path,
            stats=training_stats,
            workers=generator_workers
        )

    callbacks = [
        checkpoint, 
        performance_callback,
        TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_reduce_lr,
                          verbose=1, mode="min", min_lr=min_lr),
        EarlyStopping(monitor="val_loss",min_delta=0,patience=patience_early_stop, verbose=0,
                        mode="min",baseline=None,restore_best_weights=False,),
    ]

    print("** start training **")

    history = model.fit_generator(
        generator=train_sequence,
        initial_epoch=current_epoch,
        epochs=epochs,
        class_weight=class_weights,
        validation_data=validation_sequence,
        callbacks=callbacks,
        workers=generator_workers,
        shuffle=False,
    )

    # dump history
    print("** dump history **")
    with open(os.path.join(output_dir, "history.pkl"), "wb") as f:
        pickle.dump({
            "history": history.history
        }, f)
    print("** done! **")

if __name__ == "__main__":
    main()