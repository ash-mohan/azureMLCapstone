import json
import pandas as pd
import joblib
from azureml.core.model import Model
import os

def init():
    global model
    
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

def run(raw_data):  
    data = pd.read_json(raw_data, orient="records")
    predictions = model.predict(data)
    return {
        "predictions": predictions.tolist()
    }