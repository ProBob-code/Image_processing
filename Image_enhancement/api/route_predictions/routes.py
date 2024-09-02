from crypt import methods
import numpy as np
from flask import Blueprint, request, jsonify
import pickle
from class_api_response import ApiResponse

# Define Blueprint
prediction_bp = Blueprint('prediction_bp', __name__)

# Load the model
model = pickle.load(open('../models/model_salary_predictor.pkl','rb'))

@prediction_bp.route('/salary', methods=['POST'])
def salarypredict():
    
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['exp'])]])

    # Take the first value of prediction
    output = prediction[0]

    apiResponse = ApiResponse()
    return apiResponse.responseSuccess("Salary predicted successfully!", { "experience": data['exp'], "prediction": output })