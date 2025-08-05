import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load fertilizer dataset
df = pd.read_csv("data_core.csv")

# Label encoding
label_encoders = {}
for col in ['Soil_Type', 'Crop', 'FertilizerName']:  # ✅ Include 'Crop' and 'Soil_Type'
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df[['Temperature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop', 'Nitrogen', 'Potassium', 'Phosphorus']]
y = df['FertilizerName']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'fertilizer_model.pkl')
joblib.dump(label_encoders, 'fertilizer_label_encoders.pkl')

# Optional: Accuracy
print("✅ Fertilizer Model Accuracy:", model.score(X_test, y_test))
