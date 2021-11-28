#!/usr/bin/env python3

from flask import Flask, request, Response
import pickle
import pandas as pd
import json

app = Flask(__name__)

# Load the model
model = pickle.load(open("Lasso_model.pkl", 'rb'))

@app.route("/bmi", methods=['POST'])
def predict_bmi_():
    # 1) Read data from JSON to dict
    payload = request.get_json(force=True)
    # 2) Convert dict to Pandas df
    df = pd.DataFrame.from_dict(payload, orient="index").T
    data = df.drop(labels=["BMI", "ID"], axis=1, errors='ignore')
    print(f"Debugging df: {data}")
    # 4) predict BMI with the model
    prediction = model.predict(data)
    print(f"Debugging prediction: {prediction}")
    # 5) Return prediction as json
    return json.dumps(prediction.tolist())