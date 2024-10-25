import time
import requests
import random


FLASK_APP_URL = "http://127.0.0.1:5000/predict"


THINGSPEAK_API_KEY = 'HDU7CQTSZKKP7CK2'


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

def fetch_from_thingspeak(sensor_data):
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
        
        sensor_data = {
            'N': random.uniform(0, 100), 
            'P': random.uniform(0, 100), 
            'K': random.uniform(0, 100), 
            'temperature': random.uniform(15, 40),  
            'humidity': random.uniform(30, 90), 
            'ph': random.uniform(5.0, 9.0),  
            'rainfall': random.uniform(30, 90), 
        }

    
        send_to_flask(sensor_data)

        fetch_from_thingspeak(sensor_data)

       
        time.sleep(2)  

if __name__ == "__main__":
    simulate_sensor_data()
