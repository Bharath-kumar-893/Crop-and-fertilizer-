import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('ensemble_crop_model.pkl')
encoders = joblib.load('crop_label_encoders.pkl')

# Sample input
sample_input = {
    'Nitrogen': 69.07,
    'Phosphorus': 53.95,
    'Potassium': 88.06,
    'Temperature': 17.26,
    'Humidity': 72.94,
    'pH_Value': 4.63,
    'Rainfall': 302.84
}

# Convert to DataFrame
input_df = pd.DataFrame([sample_input])

# Predict (encoded)
predictions = model.predict(input_df)[0]

# Decode predictions
crop_decoded = encoders['Crop'].inverse_transform([predictions[0]])[0]
soil_decoded = encoders['Soil_Type'].inverse_transform([predictions[1]])[0]
variety_decoded = encoders['Variety'].inverse_transform([predictions[2]])[0]

# Output
print("✅ Crop Model Test")
print(f"Predicted Crop      : {crop_decoded}")
print(f"Suggested Soil Type : {soil_decoded}")
print(f"Recommended Variety : {variety_decoded}")
