import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
import cv2
from skimage.transform import resize
import time

class AugmentedImageSequence(Sequence):
    def __init__(self, dataset_csv_file, base_dir, batch_size, target_size, augmenter, steps, shuffle_on_epoch_end, random_state):
        self.dataset_df = pd.read_csv(dataset_csv_file)
        self.base_dir = base_dir

        self.batch_size = batch_size
        self.target_size = target_size
        self.augmenter = augmenter
        
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        
        self.prepare_dataset()

        if steps is None:
            self.steps = int(np.ceil(len(self.x) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx): 
        x_batch_files = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch_files = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]       
        
        batch_x = self.load_x(x_batch_files)
        batch_y = self.load_y(y_batch_files)

        return batch_x, batch_y

    def load_x(self, files):
        pass
    
    def load_y(self, files):
        pass

    def normalize(self, img):
        pass
    
    def prepare_dataset(self):
        pass

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()

class segmentation_gen(AugmentedImageSequence):
    def __init__(self, dataset_csv_file, base_dir, batch_size=16, target_size=(256, 256), augmenter=None, steps=None,
                shuffle_on_epoch_end=True, random_state=1):

        super(segmentation_gen, self).__init__(dataset_csv_file, base_dir, batch_size, target_size, augmenter, steps, shuffle_on_epoch_end, random_state)

    def __set_aug(self):
        self.det_aug = self.augmenter.to_deterministic()

    def load_x(self, files):
        # f[1] -> folder, f[0] -> filename
        file_path = [os.path.join(self.base_dir, f[1], f[0]) for f in files]
        #read
        image_array = [cv2.imread(f, 0) for f in file_path]
        #resize and normalize
        image_array = np.array([self.normalize(resize(img, self.target_size)) for img in image_array])

        #get a deterministic augmenter to do same augmentation on both image and mask
        if(self.augmenter is not None):
            self.__set_aug()
            image_array = np.array(self.det_aug.augment_images(image_array))
        image_array = np.expand_dims(image_array, -1)
        return image_array

    def load_y(self, files):
        file_path = [os.path.join(self.base_dir, f[1], f[0]) for f in files]
        #read
        mask_array = [cv2.imread(f, 0) for f in file_path]
        #resize and thresholding
        mask_array = np.array([resize(img, self.target_size) for img in mask_array])
        
        if(self.augmenter is not None):
            mask_array = np.array(self.det_aug.augment_images(mask_array))
        mask_array = np.expand_dims(mask_array, -1)

        return mask_array

    def normalize(self, img):
        img = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)) 
        #Histogram equaliztion of the given image
        img = cv2.equalizeHist(img) 
        #Gamma correction with gamma = 0.5
        gamma = 0.5
        table = np.array([(np.float_power(i/255.0, gamma)) * 255 for i in np.arange(0, 256)]).astype("float32")
        img = cv2.LUT(img, table)

        return (img/255.0)
        
    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1.0, random_state=self.random_state)
        self.x = np.concatenate([df['filename'].values.reshape(-1, 1), df['image_folder'].values.reshape(-1, 1)], axis = -1)
        
        self.y = np.concatenate([df['filename'].values.reshape(-1, 1), df['mask_folder'].values.reshape(-1, 1)], axis = -1)

        
class classification_gen(AugmentedImageSequence):
    def __init__(self, dataset_csv_file, base_dir, mask_folder, class_names, N, normalization_func=None, 
                batch_size=16, target_size=(1024, 1024), patch_size=(244,244), augmenter=None, steps=None, 
                shuffle_on_epoch_end=True, random_state=1):

        self.class_names = class_names
        self.normalization_func = normalization_func
        self.mask_folder = mask_folder
        self.patch_size = patch_size
        super(classification_gen, self).__init__(dataset_csv_file, base_dir, batch_size, target_size, augmenter, steps, shuffle_on_epoch_end, random_state)
        
        self.half_patch_size = (int(patch_size[0]/2.0), int(patch_size[1]/2.0))
        self.upper_limit = (self.target_size[0] - self.half_patch_size[0], self.target_size[1] - self.half_patch_size[1])
        self.lower_limit = self.half_patch_size
        self.N = N

    def load_x(self, files):
        # f[1] -> folder, f[0] -> filename
        file_path = [os.path.join(self.base_dir, "classification images", f[0]) for f in files]
        
        #read
        image_array = [cv2.imread(f) for f in file_path]
       
        #resize and normalize
        image_array = np.array([resize(img, self.target_size) for img in image_array])
        image_array = np.array([self.normalize(img) for img in image_array])
        
        #mask the images
        mask_array = self.__get_masks(files)
        image_array = image_array * np.expand_dims(mask_array, -1)

        #select a random 224x224 patch
        patch_array = []
        for img, mask in zip(image_array, mask_array):
            
            patches = np.empty((self.N, self.patch_size[0], self.patch_size[1], 3))
            for i in range(self.N):
                patch = self.generate_patch(img, mask)

                if(self.normalization_func is not None):
                    patch = self.normalization_func(patch) 
                patches[i] = patch

            if(self.augmenter is not None):
                patches = np.array(self.augmenter.augment_images(patches)) 

            patch_array.append(patches)

        return np.asarray(patch_array)
    
    def __get_masks(self, files):
        file_path = [os.path.join(self.base_dir, self.mask_folder, f[0]) for f in files]
        #read
        mask_array = [cv2.imread(f, 0) for f in file_path] 
        #resize
        mask_array = np.array([resize(img, self.target_size)  for img in mask_array])
        
        return mask_array

    def load_y(self, files):
        return files

    def get_y_true(self):
        if self.shuffle:
            raise ValueError("You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.")
        return self.y[:self.steps*self.batch_size, :]

    def normalize(self, img):
        img = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)) 
        #Histogram equaliztion of the given image
        for i in range(img.shape[-1]):
          img[:, :, i] = cv2.equalizeHist(img[:, :, i]) 
        #Gamma correction with gamma = 0.5
        gamma = 0.5
        table = np.array([(np.float_power(i/255.0, gamma)) * 255 for i in np.arange(0, 256)]).astype("float32")
        img = cv2.LUT(img, table)

        return (img/255.0)
    
    def generate_patch(self, img, mask):
        #select a random 224x224 patch  
        while(True):
            x, y = np.random.randint(self.lower_limit[0], self.upper_limit[0]), np.random.randint(self.lower_limit[1], self.upper_limit[1])
            if(mask[x, y] > 0.1):   
                patch = img[x-self.half_patch_size[0]:x+self.half_patch_size[0], y-self.half_patch_size[1]:y+self.half_patch_size[1]]
                break        
        return patch

    def prepare_dataset(self):
        np.random.RandomState(seed=int(time.time()))

        df = self.dataset_df.sample(frac=1.0, random_state=self.random_state)
        self.x = np.concatenate([df['filename'].values.reshape(-1, 1), df['image_folder'].values.reshape(-1, 1)], axis = -1)      
        self.y = df[self.class_names].astype('int32').values
