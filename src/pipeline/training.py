import os
import sys
import tensorflow as tf
import warnings 
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
)

from src.components.data_loader import DataGenerator, DataLoaderConfig
from src.components.data_transformation import DataAugmentation
from src.logger import logging
from src.exception import CustomException
warnings.filterwarnings("ignore")

# HYPERPARAMETERS
LEARNING_RATE = 1e-4
SAVE_DIR = "./models"
MODEL_NAME = "vgg16_flower_classifier.h5"
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TARGET_SIZE = (256, 256)
EPOCHS = 50


def train():
    # Data loading 
    directories = DataLoaderConfig()
    train_dir = directories.train_dir
    val_dir = directories.val_dir

    # Data augmenter
    augmenter = DataAugmentation(target_size=TARGET_SIZE)

    train_datagen = DataGenerator(
        image_dir=train_dir,
        batch_size=TRAIN_BATCH_SIZE,
        target_size=TARGET_SIZE,
        augmenter=augmenter,
        shuffle=True
    )
    
    validation_datagen = DataGenerator(
        image_dir=val_dir,
        batch_size=VAL_BATCH_SIZE,
        target_size=TARGET_SIZE,
        augmenter=None,
        shuffle=False
    )

    print("Data Loaded successfully.")

    # Model
    num_classes = len(train_datagen.classes_name)
    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    for layer in vgg16.layers:
        layer.trainable = False

    def ModelTop(bottom_model, num_classes):
        x = bottom_model.output
        x = Flatten(name="flatten") (x)
        x = Dense(512, activation="relu") (x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation="relu") (x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation="relu") (x)
        x = Dense(num_class, activation="softmax") (x)
        return x

    Fc_head = ModelTop(vgg16, num_classes)
    vgg16 = Model(inputs = vgg16.input, outputs = Fc_Head)
    # Loss and optimizer
    optimizer = RMSprop(learning_rate=LEARNING_RATE)
    loss = "sparse_categorical_crossentropy"

    vgg16.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Checkpoint:
    
    # Callbacks:
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", patience=2, verbose=1),
    ]
    
    # Training 
    history = vgg16.fit(
        train_datagen,
        validation_data=validation_datagen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print("Model Trained successfully.")

    # Save Model:
    logging.info(f"Saving trained model in {SAVE_DIR} directory")
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, MODEL_NAME)
    vgg16.save(save_path)
    print(f"Model saved successfully at : {save_path}")

    # Status of the Model :
    print("="*50)
    print("Model Status: \n")
    print("Epochs : " + str(len(history.history["loss"])))
    print("Final Training Loss : " + str(history.history["loss"][-1]))
    print("Final Validation Loss : " + str(history.history["val_loss"][-1]))
    print("Min. Training Loss : " + str(min(history.history["loss"])))
    print("Min. Validation Loss : " + str(min(history.history["val_loss"])))
    print("="*50)
    print("\n")

    return history
    
if __name__ == "__main__":
    history = train()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title('Training Loss')
    ax[0].plot(history.history['loss'])
    ax[1].set_title('Validation Loss')
    ax[1].plot(history.history['val_loss'])
    plt.savefig('loss_plot.png')
    plt.show()