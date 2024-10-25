import requests
import pandas as pd
from flask import Flask, jsonify, render_template, request
import numpy as np
import pickle
import os
from dotenv import load_dotenv



# Now you can use ADAFRUIT_IO_KEY in your code

app = Flask(__name__)

historical_data = pd.read_csv('historical_crops_data.csv')
model = pickle.load(open('crop_prediction_model.pkl', 'rb'))

# Adafruit IO credentials
load_dotenv()  # Load environment variables from .env file
ADAFRUIT_IO_KEY = os.getenv("ADAFRUIT_IO_KEY")

ADAFRUIT_IO_USERNAME = 'soilnutrient0005'
FEED_KEY = 'sensor-data'  # Use the key for the combined data feed

def get_combined_sensor_data():
    url = f"https://io.adafruit.com/api/v2/{ADAFRUIT_IO_USERNAME}/feeds/{FEED_KEY}/data/last"
    headers = {'X-AIO-Key': ADAFRUIT_IO_KEY}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data_string = response.json()['value']
        print(f"Raw sensor data string: {data_string}")  # Debugging
        # Parse the string (assuming format is "M: 0 N: 0 P: 0 K: 0 T: 30.20 H: 51.00")
        data_parts = data_string.split()
        sensor_data = {
            'M': float(data_parts[1]),
            'N': float(data_parts[3]),
            'P': float(data_parts[5]),
            'K': float(data_parts[7]),
            'temperature': float(data_parts[9]),
            'humidity': float(data_parts[11])
        }
        return sensor_data
    else:
        raise Exception(f"Failed to fetch data from Adafruit.io: {response.status_code}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'GET':
            # Fetch real-time sensor data from Adafruit.io
            sensor_data = get_combined_sensor_data()
        else:
            # If it's a POST request, retrieve data from the form
            sensor_data = {
                'N': float(request.form['N']),
                'P': float(request.form['P']),
                'K': float(request.form['K']),
                'temperature': float(request.form['temperature']),
                'humidity': float(request.form['humidity']),
                'M': float(request.form['rainfall'])  # Using 'rainfall' for soil moisture
            }
        
        season = request.form.get('season', 'Kharif')  # Get the season from the form or default to 'Kharif'

        # Prepare data for prediction
        final_features = np.array([[sensor_data['N'], sensor_data['P'], sensor_data['K'], sensor_data['temperature'], sensor_data['humidity'], sensor_data['M']]])
        prediction = model.predict(final_features)
        predicted_crop = prediction[0]

        # Compare with historical data for the selected season
        historical_crops = historical_data[historical_data['Season'] == season]
        historically_grown = historical_crops['Crop'].unique() if not historical_crops.empty else []
        
        comparison_result = f"The predicted crop '{predicted_crop}' {'has' if predicted_crop in historically_grown else 'has NOT'} been historically grown in this period."
        historically_grown_text = f'Crops historically grown in {season}: {", ".join(historically_grown)}'

        response = {
            'sensor_data': sensor_data,
            'predicted_crop': predicted_crop,
            'comparison_text': comparison_result,
            'historical_grown_text': historically_grown_text
        }

        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
