import os
import sys
import numpy as np
import tensorflow as tf
import warnings

from PIL import Image
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
warnings.filterwarnings("ignore")

@dataclass
class DataLoaderConfig:
    logging.info("Found Data Loader configs")
    train_dir : str = os.path.join("artifacts", "train") 
    val_dir : str = os.path.join("artifacts", "validation")


class DataGenerator(Sequence):
    def __init__(self,
                image_dir=None,
                batch_size=1,
                target_size=(256, 256),
                augmenter=None,
                shuffle=False):
        
        super().__init__()
        
        logging.info("Creating a Data Generator")
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augmenter = augmenter
        self.image_paths, self.labels = self._load_data()
        self.classes_name = os.listdir(self.image_dir)
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        self.on_epoch_end()

    def _load_data(self):
        """Loads image data paths and their corresponding labels"""
        try:
            image_paths = []
            labels = []
            for class_name in os.listdir(self.image_dir):
                class_dir = os.path.join(self.image_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        image_paths.append(img_path)
                        labels.append(class_name)
            return image_paths, labels
        except Exception as e:
            raise CustomException(e, sys)

    def __len__(self):
        """Returns the number of batches of same batchsize"""
        try:
            return int(np.floor(len(self.image_paths) / self.batch_size))
        except Exception as e:
            raise CustomException(e, sys)    
    
    def __getitem__(self, index):
        try:
            start = index * self.batch_size
            end = (index + 1) * self.batch_size
            batch_images = self.image_paths[start : end]
            batch_labels = self.labels[start : end]

            images = []
            for image in batch_images:
                img = Image.open(image).convert("RGB")
                if not self.augmenter:
                    img = img.resize(self.target_size) 
                    img = np.asarray(img) / 255
                else:
                    img = self.augmenter.augment(img) 
                images.append(img)
            
            return np.array(images), np.array(batch_labels)
        except Exception as e:
            raise CustomException(e, sys)

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        try:
            if self.shuffle is True:
                combined = list(zip(self.image_paths, self.labels))
                np.random.shuffle(combined)
                self.image_paths, self.labels = zip(*combined)
        except Exception as e:
            raise CustomException(e, sys)