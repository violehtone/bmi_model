#!/usr/bin/env python3
import pickle
import pandas as pd
import json
from flask import Flask, request

app = Flask(__name__)

# Load the model
MODEL_FILENAME = "src/Lasso_model.pkl"

# List of the required features for the prediction
REQUIRED_FEATURES = [
    'VMALE', 'AGE', 'BMI32_WGRS', 'BMI32_GRS', 'XXL_VLDL_P', 'XXL_VLDL_L', 'XXL_VLDL_PL', 'XXL_VLDL_C', 'XXL_VLDL_CE',
    'XXL_VLDL_FC', 'XXL_VLDL_TG', 'XL_VLDL_P', 'XL_VLDL_L', 'XL_VLDL_PL', 'XL_VLDL_C', 'XL_VLDL_CE', 'XL_VLDL_FC',
    'XL_VLDL_TG', 'L_VLDL_P', 'L_VLDL_L', 'L_VLDL_PL', 'L_VLDL_C', 'L_VLDL_CE', 'L_VLDL_FC', 'L_VLDL_TG', 'M_VLDL_P',
    'M_VLDL_L', 'M_VLDL_PL', 'M_VLDL_C', 'M_VLDL_CE', 'M_VLDL_FC', 'M_VLDL_TG', 'S_VLDL_P', 'S_VLDL_L', 'S_VLDL_PL',
    'S_VLDL_C', 'S_VLDL_CE', 'S_VLDL_FC', 'S_VLDL_TG', 'XS_VLDL_P', 'XS_VLDL_L', 'XS_VLDL_PL', 'XS_VLDL_C',
    'XS_VLDL_CE', 'XS_VLDL_FC', 'XS_VLDL_TG', 'IDL_P', 'IDL_L', 'IDL_PL', 'IDL_C', 'IDL_CE', 'IDL_FC', 'IDL_TG',
    'L_LDL_P', 'L_LDL_L', 'L_LDL_PL', 'L_LDL_C', 'L_LDL_CE', 'L_LDL_FC', 'L_LDL_TG', 'M_LDL_P', 'M_LDL_L', 'M_LDL_PL',
    'M_LDL_C', 'M_LDL_CE', 'M_LDL_FC', 'M_LDL_TG', 'S_LDL_P', 'S_LDL_L', 'S_LDL_PL', 'S_LDL_C', 'S_LDL_CE', 'S_LDL_FC',
    'S_LDL_TG', 'XL_HDL_P', 'XL_HDL_L', 'XL_HDL_PL', 'XL_HDL_C', 'XL_HDL_CE', 'XL_HDL_FC', 'XL_HDL_TG', 'L_HDL_P',
    'L_HDL_L', 'L_HDL_PL', 'L_HDL_C', 'L_HDL_CE', 'L_HDL_FC', 'L_HDL_TG', 'M_HDL_P', 'M_HDL_L', 'M_HDL_PL', 'M_HDL_C',
    'M_HDL_CE', 'M_HDL_FC', 'M_HDL_TG', 'S_HDL_P', 'S_HDL_L', 'S_HDL_PL', 'S_HDL_C', 'S_HDL_CE', 'S_HDL_FC', 'S_HDL_TG',
    'XXL_VLDL_PL_PER', 'XXL_VLDL_C_PER', 'XXL_VLDL_CE_PER', 'XXL_VLDL_FC_PER', 'XXL_VLDL_TG_PER', 'XL_VLDL_PL_PER',
    'XL_VLDL_C_PER', 'XL_VLDL_CE_PER', 'XL_VLDL_FC_PER', 'XL_VLDL_TG_PER', 'L_VLDL_PL_PER', 'L_VLDL_C_PER',
    'L_VLDL_CE_PER', 'L_VLDL_FC_PER', 'L_VLDL_TG_PER', 'M_VLDL_PL_PER', 'M_VLDL_C_PER', 'M_VLDL_CE_PER',
    'M_VLDL_FC_PER', 'M_VLDL_TG_PER', 'S_VLDL_PL_PER', 'S_VLDL_C_PER', 'S_VLDL_CE_PER', 'S_VLDL_FC_PER',
    'S_VLDL_TG_PER', 'XS_VLDL_PL_PER', 'XS_VLDL_C_PER', 'XS_VLDL_CE_PER', 'XS_VLDL_FC_PER', 'XS_VLDL_TG_PER',
    'IDL_PL_PER', 'IDL_C_PER', 'IDL_CE_PER', 'IDL_FC_PER', 'IDL_TG_PER', 'L_LDL_PL_PER', 'L_LDL_C_PER', 'L_LDL_CE_PER',
    'L_LDL_FC_PER', 'L_LDL_TG_PER', 'M_LDL_PL_PER', 'M_LDL_C_PER', 'M_LDL_CE_PER', 'M_LDL_FC_PER', 'M_LDL_TG_PER',
    'S_LDL_PL_PER', 'S_LDL_C_PER', 'S_LDL_CE_PER', 'S_LDL_FC_PER', 'S_LDL_TG_PER', 'XL_HDL_PL_PER', 'XL_HDL_C_PER',
    'XL_HDL_CE_PER', 'XL_HDL_FC_PER', 'XL_HDL_TG_PER', 'L_HDL_PL_PER', 'L_HDL_C_PER', 'L_HDL_CE_PER', 'L_HDL_FC_PER',
    'L_HDL_TG_PER', 'M_HDL_PL_PER', 'M_HDL_C_PER', 'M_HDL_CE_PER', 'M_HDL_FC_PER', 'M_HDL_TG_PER', 'S_HDL_PL_PER',
    'S_HDL_C_PER', 'S_HDL_CE_PER', 'S_HDL_FC_PER', 'S_HDL_TG_PER', 'VLDL_D', 'LDL_D', 'HDL_D', 'SERUM_C', 'VLDL_C',
    'REMNANT_C', 'LDL_C', 'HDL_C', 'HDL2_C', 'HDL3_C', 'ESTC', 'FREEC', 'SERUM_TG', 'VLDL_TG', 'LDL_TG', 'HDL_TG',
    'TOTPG', 'TG.PG', 'PC', 'SM', 'TOTCHO', 'APOA1', 'APOB', 'APOB.APOA1', 'TOTFA', 'UNSAT', 'DHA', 'LA', 'FAW3',
    'FAW6', 'PUFA', 'MUFA', 'SFA', 'DHA.FA', 'LA.FA', 'FAW3.FA', 'FAW6.FA', 'PUFA.FA', 'MUFA.FA', 'SFA.FA', 'GLC',
    'LAC', 'PYR', 'CIT', 'GLOL', 'ALA', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'VAL', 'PHE', 'TYR', 'ACE', 'ACACE',
    'BOHBUT', 'CREA', 'ALB', 'GP']


@app.route("/", methods=['GET'])
def ping():
    """Ping the server to verify that it is working."""
    return "Ping!"


@app.route("/bmi", methods=['POST'])
def predict_bmi():
    """Predict a person's BMI from a JSON"""

    # Load the lasso regression model
    lasso_model = pickle.load(open(MODEL_FILENAME, 'rb'))

    # Read data from JSON and convert to properly formatted pandas dataframe
    payload = request.get_json(force=True)
    df = pd.DataFrame.from_dict(payload, orient="index").T

    # Return a bad request (400) if the request body don't have all the required features to make the prediction
    if not all(feature in list(df.columns.values) for feature in REQUIRED_FEATURES):
        missing_features = list(set(REQUIRED_FEATURES) - set(list(df.columns.values)))
        return f"One (or more) of the required features are missing. The missing features are: {missing_features}", 400

    # predict BMI with the model
    prediction = lasso_model.predict(df)
    return json.dumps({"bmi": prediction[0]}), 200


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
