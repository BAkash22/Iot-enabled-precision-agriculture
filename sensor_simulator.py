import time
import requests
import random

# Flask app URL for the /predict route
FLASK_APP_URL = "http://127.0.0.1:5000/predict"

# Your ThingSpeak Write API key (replace this with your actual ThingSpeak API key)
THINGSPEAK_API_KEY = 'HDU7CQTSZKKP7CK2'

# ThingSpeak URL for sending data
THINGSPEAK_URL = 'https://api.thingspeak.com/update'

def send_to_flask(sensor_data):
    """Send the simulated sensor data to the Flask app."""
    try:
        response = requests.post(FLASK_APP_URL, json=sensor_data)
        if response.status_code == 200:
            print(f"Data sent to Flask app: {sensor_data}")
            print(f"Flask app response: {response.json()}")
        else:
            print(f"Failed to send data to Flask app, status code: {response.status_code}")
    except Exception as e:
        print(f"Error connecting to Flask app: {e}")

def send_to_thingspeak(sensor_data):
    """Send the sensor data to ThingSpeak."""
    try:
        response = requests.get(THINGSPEAK_URL, params={
            'api_key': THINGSPEAK_API_KEY,
            'field1': sensor_data['N'],
            'field2': sensor_data['P'],
            'field3': sensor_data['K'],
            'field4': sensor_data['temperature'],
            'field5': sensor_data['humidity'],
            'field6': sensor_data['ph'],
            'field7': sensor_data['rainfall']
        })

        if response.status_code == 200:
            print(f"Data sent to ThingSpeak: {sensor_data}")
        else:
            print(f"Failed to send data to ThingSpeak, status code: {response.status_code}")
    except Exception as e:
        print(f"Error connecting to ThingSpeak: {e}")

def simulate_sensor_data():
    """Simulate sensor data and send it to the Flask app and ThingSpeak."""
    while True:
        # Simulate random sensor values
        sensor_data = {
            'N': random.uniform(0, 100),  # Simulating Nitrogen levels
            'P': random.uniform(0, 100),  # Simulating Phosphorus levels
            'K': random.uniform(0, 100),  # Simulating Potassium levels
            'temperature': random.uniform(15, 40),  # Simulating temperature in °C
            'humidity': random.uniform(30, 90),  # Simulating humidity percentage
            'ph': random.uniform(5.0, 9.0),  # Simulating soil pH levels
            'rainfall': random.uniform(30, 90),  # Simulating rainfall in mm
        }

        # Send data to Flask app
        send_to_flask(sensor_data)

        # Send data to ThingSpeak
        send_to_thingspeak(sensor_data)

        # Wait for a few seconds before sending the next set of data
        time.sleep(2)  # Adjust this delay as necessary

if __name__ == "__main__":
    simulate_sensor_data()
