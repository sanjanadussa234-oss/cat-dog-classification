import os
import mlflow
import mlflow.tensorflow
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------------
# SETTINGS
# -----------------------------------

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# -----------------------------------
# DATASET PATHS
# -----------------------------------

train_dir = "dataset/train"
val_dir = "dataset/validation"

# -----------------------------------
# IMAGE GENERATORS
# -----------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# -----------------------------------
# MLFLOW EXPERIMENT
# -----------------------------------
mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment("Cats_vs_Dogs_Classification")

with mlflow.start_run():

    # Log Parameters
    mlflow.log_param("Image Size", IMG_SIZE)
    mlflow.log_param("Batch Size", BATCH_SIZE)
    mlflow.log_param("Epochs", EPOCHS)
    mlflow.log_param("Learning Rate", LEARNING_RATE)

    # -----------------------------------
    # LOAD MOBILENETV2
    # -----------------------------------

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    # -----------------------------------
    # BUILD MODEL
    # -----------------------------------

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # -----------------------------------
    # COMPILE MODEL
    # -----------------------------------

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # -----------------------------------
    # TRAIN MODEL
    # -----------------------------------

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # -----------------------------------
    # CREATE DIRECTORIES
    # -----------------------------------

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # -----------------------------------
    # SAVE MODEL
    # -----------------------------------

    model.save("models/model.keras")

    print("\nModel Saved Successfully!")

    # -----------------------------------
    # SAVE HISTORY
    # -----------------------------------

    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        "reports/training_history.csv",
        index=False
    )

    # -----------------------------------
    # LOG METRICS
    # -----------------------------------

    mlflow.log_metric(
        "Final Training Accuracy",
        history.history['accuracy'][-1]
    )

    mlflow.log_metric(
        "Final Validation Accuracy",
        history.history['val_accuracy'][-1]
    )

    mlflow.log_metric(
        "Final Training Loss",
        history.history['loss'][-1]
    )

    mlflow.log_metric(
        "Final Validation Loss",
        history.history['val_loss'][-1]
    )

    # -----------------------------------
    # ACCURACY PLOT
    # -----------------------------------

    plt.figure(figsize=(8,5))

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    accuracy_plot_path = "reports/accuracy_plot.png"

    plt.savefig(accuracy_plot_path)
    plt.close()

    # -----------------------------------
    # LOSS PLOT
    # -----------------------------------

    plt.figure(figsize=(8,5))

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    loss_plot_path = "reports/loss_plot.png"

    plt.savefig(loss_plot_path)
    plt.close()

    # -----------------------------------
    # LOG ARTIFACTS
    # -----------------------------------

    mlflow.log_artifact(accuracy_plot_path)
    mlflow.log_artifact(loss_plot_path)
    mlflow.log_artifact("reports/training_history.csv")

    # -----------------------------------
    # LOG MODEL
    # -----------------------------------

    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model"
    )

    print("\nMLflow Logging Completed!")
    print("\nTraining Completed Successfully!")