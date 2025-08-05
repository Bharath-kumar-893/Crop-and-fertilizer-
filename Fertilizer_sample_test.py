import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('fertilizer_model.pkl')
encoders = joblib.load('fertilizer_label_encoders.pkl')

# Sample input (change based on your dataset)
sample_input = {
    'Temperature': 26,
    'Humidity': 75,
    'Moisture': 40,
    'Soil_Type': 'Loamy',
    'Crop_Type': 'Wheat',
    'Nitrogen': 50,
    'Potassium': 30,
    'Phosphorus': 40
}

# Encode Soil_Type and Crop_Type
try:
    sample_input['Soil_Type'] = encoders['Soil_Type'].transform([sample_input['Soil_Type']])[0]
    sample_input['Crop_Type'] = encoders['Crop_Type'].transform([sample_input['Crop_Type']])[0]
except ValueError as e:
    print("❌ Encoding Error:", e)
    exit()

# Prepare input list
features = [
    sample_input['Temperature'],
    sample_input['Humidity'],
    sample_input['Moisture'],
    sample_input['Soil_Type'],
    sample_input['Crop_Type'],
    sample_input['Nitrogen'],
    sample_input['Potassium'],
    sample_input['Phosphorus']
]

# Predict and decode
predicted_code = model.predict([features])[0]
fertilizer_name = encoders['FertilizerName'].inverse_transform([predicted_code])[0]

# Output
print("✅ Fertilizer Model Test")
print(f"Recommended Fertilizer: {fertilizer_name}")
