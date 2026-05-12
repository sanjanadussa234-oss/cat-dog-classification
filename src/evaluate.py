import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# -----------------------------------
# SETTINGS
# -----------------------------------

IMG_SIZE = 224
BATCH_SIZE = 32

# -----------------------------------
# TEST DATA PATH
# -----------------------------------

test_dir = "dataset/test"

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = load_model("models/model.keras")

print("\nModel Loaded Successfully!")

# -----------------------------------
# TEST GENERATOR
# -----------------------------------

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# -----------------------------------
# MAKE PREDICTIONS
# -----------------------------------

pred_probs = model.predict(test_generator)

predictions = (pred_probs > 0.5).astype(int)

# Flatten predictions
predictions = predictions.flatten()

# -----------------------------------
# TRUE LABELS
# -----------------------------------

true_labels = test_generator.classes

# -----------------------------------
# CLASS NAMES
# -----------------------------------

class_names = list(test_generator.class_indices.keys())

# -----------------------------------
# ACCURACY
# -----------------------------------

accuracy = accuracy_score(true_labels, predictions)

print(f"\nTest Accuracy: {accuracy:.4f}")
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy"],
    "Value": [accuracy]
})

metrics_df.to_csv(
    "reports/evaluation_metrics.csv",
    index=False
)

# -----------------------------------
# CLASSIFICATION REPORT
# -----------------------------------

report = classification_report(
    true_labels,
    predictions,
    target_names=class_names
)

print("\nClassification Report:\n")
print(report)

# -----------------------------------
# SAVE REPORT
# -----------------------------------

os.makedirs("reports", exist_ok=True)

with open("reports/classification_report.txt", "w") as f:
    f.write(report)

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------

cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("reports/confusion_matrix.png")

plt.show()

print("\nEvaluation Completed Successfully!")