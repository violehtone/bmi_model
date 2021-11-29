#!/usr/bin/env python3

from flask import Flask, request, Response
import pickle
import pandas as pd
import json

MODEL_FILENAME = "Lasso_model.pkl"
app = Flask(__name__)

# Load the model
model = pickle.load(open(MODEL_FILENAME, 'rb'))

@app.route("/", methods=['GET'])
def ping():
    return "API is running!"

@app.route("/bmi", methods=['POST'])
def predict_bmi():
    # 1) Read data from JSON to dict
    payload = request.get_json(force=True)
    # 2) Convert dict to Pandas df and remove BMI and ID if they exist in the df
    df = pd.DataFrame.from_dict(payload, orient="index").T
    data = df.drop(labels=["BMI", "ID"], axis=1, errors='ignore')
    # 4) predict BMI with the model
    prediction = model.predict(data)
    # 5) Return prediction as json
    return json.dumps(prediction.tolist())


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')