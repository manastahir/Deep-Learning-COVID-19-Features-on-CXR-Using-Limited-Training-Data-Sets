import json
import keras.backend as kb
import numpy as np
import os
import shutil
import warnings
import cv2
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import jaccard_score

class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model
    """
    def __init__(self, sequence, class_names, weights_path, stats=None, workers=1):
        super(Callback, self).__init__()
        self.sequence = sequence
        self.workers = workers
        self.class_names = class_names
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            f"best_{os.path.split(weights_path)[1]}",
        )
        self.auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "auroc.log",
        )
        self.stats_output_path = os.path.join(
            os.path.split(weights_path)[0],
            ".training_stats.json"
        )
        # for resuming previous training
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}

        # aurocs log
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.
        """
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        self.stats["epoch"] = epoch
        print(f"current learning rate: {self.stats['lr']}")

        y_hat = self.model.predict_generator(self.sequence, workers=self.workers)
        y = self.sequence.get_y_true()
        
        print(f"*** epoch#{epoch + 1} val auroc ***")
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print(f"{i+1}. {self.class_names[i]}: {score}")
        print("*********************************")
        accuracy = accuracy_score(np.argmax(y, axis=1), np.argmax(y_hat, axis=1))
        # customize your multiple class metrics here
        mean_auroc = np.mean(current_auroc)
        print(f"mean auroc: {mean_auroc}")
        print(f"ACCURACY: {accuracy}")

        #update log file
        print(f"update log file: {self.auroc_log_path}")
        with open(self.auroc_log_path, "a") as f:
            f.write(f"(epoch#{epoch + 1}) auroc: {mean_auroc}, lr: {self.stats['lr']}\n")

        if mean_auroc > self.stats["best_mean_auroc"]:
            print(f"update best auroc from {self.stats['best_mean_auroc']} to {mean_auroc}")

            # 1. copy best model
            shutil.copy(self.weights_path, self.best_weights_path)

            # 2. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            print(f"update model file: {self.weights_path} -> {self.best_weights_path}")
            self.stats["best_mean_auroc"] = mean_auroc
            print("*********************************")
        return


class MultiGPUModelCheckpoint(Callback):
    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(Callback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)

class Jaccard(Callback):
    """
    Monitor Jaccard score and update model
    """
    def __init__(self, sequence, weights_path, stats=None, workers=1):
        super(Callback, self).__init__()
        self.sequence = sequence
        self.workers = workers
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            f"best_{os.path.split(weights_path)[1]}",
        )
        self.jaccard_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "jaccard.log",
        )
        self.stats_output_path = os.path.join(
            os.path.split(weights_path)[0],
            ".training_stats.json"
        )
        # for resuming previous training
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_jaccard_score": 0}

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the Jaccard Score and save the best model weights according
        to this metric.
        """
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        self.stats["epoch"] = epoch
        print(f"current learning rate: {self.stats['lr']}")

        Jaccard_score = []
        for x, y in self.sequence:
            y_hat = self.model.predict(x)
            y_hat = np.array([cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY)[1]  for img in y_hat])
            y = np.array([cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY)[1]  for img in y])
            Jaccard_score.append(jaccard_score(y.reshape(-1), y_hat.reshape(-1)))

        Jaccard_score = np.array(Jaccard_score)

        # customize your multiple class metrics here
        mean_jaccard_score = np.mean(Jaccard_score)
        print(f"mean jaccard score: {mean_jaccard_score}")

        #update log file
        print(f"update log file: {self.jaccard_log_path}")
        with open(self.jaccard_log_path, "a") as f:
            f.write(f"(epoch#{epoch + 1}) jaccard: {mean_jaccard_score}, lr: {self.stats['lr']}\n")
            
        if mean_jaccard_score > self.stats["best_mean_jaccard_score"]:
            print(f"update best Jaccard score from {self.stats['best_mean_jaccard_score']} to {mean_jaccard_score}")

            # 1. copy best model
            shutil.copy(self.weights_path, self.best_weights_path)

            # 3. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            print(f"update model file: {self.weights_path} -> {self.best_weights_path}")
            self.stats["best_mean_jaccard_score"] = mean_jaccard_score
            print("*********************************")
        return