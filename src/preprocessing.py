import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------
# IMAGE SETTINGS
# -----------------------------------

IMG_SIZE = 224
BATCH_SIZE = 32

# -----------------------------------
# DATASET PATHS
# -----------------------------------

train_dir = "dataset/train"
val_dir = "dataset/validation"
test_dir = "dataset/test"

# -----------------------------------
# DATA AUGMENTATION
# -----------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255
)

# -----------------------------------
# TRAIN GENERATOR
# -----------------------------------

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# -----------------------------------
# VALIDATION GENERATOR
# -----------------------------------

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# -----------------------------------
# TEST GENERATOR
# -----------------------------------

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# -----------------------------------
# DISPLAY INFORMATION
# -----------------------------------

print("\nClass Indices:")
print(train_generator.class_indices)

print("\nPreprocessing Completed Successfully!")