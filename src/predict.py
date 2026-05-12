import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------------
# SETTINGS
# -----------------------------------

IMG_SIZE = 224

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = load_model("models/model.keras")

print("\nModel Loaded Successfully!")

# -----------------------------------
# CLASS LABELS
# -----------------------------------

class_labels = ['cats', 'dogs']

# -----------------------------------
# IMAGE PATH
# -----------------------------------

img_path = "sample.jpg"

# -----------------------------------
# LOAD AND PREPROCESS IMAGE
# -----------------------------------

img = image.load_img(
    img_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

# -----------------------------------
# MAKE PREDICTION
# -----------------------------------

prediction = model.predict(img_array)

print("\nPrediction Score:", prediction[0][0])

# -----------------------------------
# CLASSIFY RESULT
# -----------------------------------

if prediction[0][0] > 0.5:
    predicted_class = class_labels[1]
else:
    predicted_class = class_labels[0]

print(f"\nPredicted Class: {predicted_class}")