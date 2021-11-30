import pickle
from unittest import TestCase
from src.api import app

"""
Note: These tests assume that the Lasso_model.pkl has been generated into the src directory by running the model.py
script.
"""

class TestApi(TestCase):
    # Define a test client
    CLIENT = app.test_client()

    def test_ping_successfully(self):
        """Test a successful request to the ping endpoint"""
        response = self.CLIENT.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"Ping!")

    def test_predict_bmi_with_invalid_input(self):
        """Test calling the /bmi endpoint with invalid request body"""
        response = self.CLIENT.post("/bmi", json={"VMALE": 0})
        self.assertEqual(response.status_code, 400)

    def test_predict_bmi_success(self):
        """Test a successfull request to the /bmi endpoint"""
        with open("utils/example_request_body.txt", "rb") as f:
           request_body = pickle.load(f)
        response = self.CLIENT.post("/bmi", json=request_body)
        self.assertEqual(response.status_code, 200)
