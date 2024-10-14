import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('chengalpattu_crops_dataset.csv')

# Define the features and target variable
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['crop']

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Added random_state for reproducibility
model.fit(X, y)

# Save the trained model as a pickle file
with open('crop_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Function to predict crop using the trained model
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Load the model
    with open('crop_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Make the prediction
    crop_prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    
    return crop_prediction[0]


