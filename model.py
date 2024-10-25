import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('chengalpattu_crops_dataset.csv')

# Define features and target
# We'll continue using 'rainfall' but it represents 'soil moisture'
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']]  # Treat 'rainfall' as 'soil moisture'
y = data['crop']

# Create and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
with open('crop_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Define the prediction function
def predict_crop(N, P, K, temperature, humidity, soil_moisture):  # Here, soil_moisture corresponds to 'rainfall'
    with open('crop_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Make a prediction with 6 input features
    crop_prediction = model.predict([[N, P, K, temperature, humidity, soil_moisture]])
    
    return crop_prediction[0]
