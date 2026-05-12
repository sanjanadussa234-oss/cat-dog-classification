# Image Classification System Using Deep Learning and MLOps

## Project Overview
This project is an end-to-end image classification system using MobileNetV2 with an automated MLOps pipeline.

## Features
- Image Classification using CNN
- FastAPI Inference API
- MLflow Experiment Tracking
- DVC Dataset & Model Versioning
- Tableau Dashboard
- Automated ML Pipeline

## Tech Stack
- Python
- TensorFlow
- FastAPI
- MLflow
- DVC
- Tableau

## Project Structure
```bash
src/
api/
models/
reports/
dataset/
```

## Run Training
```bash
python src/train.py
```

## Run Evaluation
```bash
python src/evaluate.py
```

## Run Prediction
```bash
python src/predict.py
```

## Run API
```bash
uvicorn api.app:app --reload
```