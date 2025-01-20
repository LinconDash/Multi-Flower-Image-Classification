import sys
import albumentations as A
import numpy as np

from PIL import Image
from src.logger import logging
from src.exception import CustomException

class DataAugmentation:
    def __init__(self, target_size=(224, 224)):
        logging.info("Creating Data Augmenter.")
        try:
            self.target_size = target_size
            self.transform = A.Compose([
                A.Resize(height=self.target_size[0], width=self.target_size[1]),  # Resize
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)
                ], p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize to ImageNet stats
            ])
        except Exception as e:
            raise CustomException(e, sys)

    def augment(self, image):
        """Applies the defined augmentation to an image."""
        try:
            augmented = self.transform(image=np.asarray(image))
            return augmented['image']
        except Exception as e:
            raise CustomException(e, sys)