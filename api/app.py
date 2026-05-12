import io
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------------
# INITIALIZE FASTAPI
# -----------------------------------

app = FastAPI(
    title="Cats vs Dogs Classification API"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = load_model("models/model.keras")

print("\nModel Loaded Successfully!")

# -----------------------------------
# SETTINGS
# -----------------------------------

IMG_SIZE = 224

class_labels = ['cats', 'dogs']

# -----------------------------------
# HOME ROUTE
# -----------------------------------

@app.get("/")
def home():

    return {
        "message": "Cats vs Dogs Classification API is Running"
    }

# -----------------------------------
# PREDICTION FUNCTION
# -----------------------------------

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# -----------------------------------
# PREDICT ROUTE
# -----------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    score = float(prediction[0][0])

    if score > 0.5:
        predicted_class = class_labels[1]
    else:
        predicted_class = class_labels[0]

    return {
        "prediction": predicted_class,
        "confidence_score": round(score, 4)
    }