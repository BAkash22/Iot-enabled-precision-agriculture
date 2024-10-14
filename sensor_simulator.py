import time
import requests
import random

# Flask app URL for the /predict route
FLASK_APP_URL = "http://127.0.0.1:5000/predict"

def simulate_sensor_data():
    while True:
        # Simulate random sensor values
        sensor_data = {
            'N': random.uniform(0, 100),  # Simulating Nitrogen levels
            'P': random.uniform(0, 100),  # Simulating Phosphorus levels
            'K': random.uniform(0, 100),  # Simulating Potassium levels
            'temperature': random.uniform(15, 40),  # Simulating temperature in °C
            'humidity': random.uniform(30, 90),  # Simulating humidity percentage
            'ph': random.uniform(5.0, 9.0),  # Simulating soil pH levels
            'rainfall': random.uniform(0, 300),  # Simulating rainfall in mm
            #'season': random.choice(['Kharif', 'Rabi', 'Zaid'])  # Simulating the season
        }

        # Send the simulated data as a POST request to the Flask app
        response = requests.post(FLASK_APP_URL, json=sensor_data)

        # Print the response from the Flask app (this is optional)
        if response.status_code == 200:
            print(f"Data sent: {sensor_data}")
            print(f"Response: {response.json()}")
        else:
            print(f"Failed to send data, status code: {response.status_code}")

        # Wait for a few seconds before sending the next set of data
        time.sleep(5)

if __name__ == "__main__":
    simulate_sensor_data()
