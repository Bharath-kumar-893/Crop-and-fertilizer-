import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# === Load Dataset ===
crop_df = pd.read_csv("sensor_Crop_Dataset.csv")

# === Encode all outputs ===
label_encoders_crop = {}
for col in ['Crop', 'Soil_Type', 'Variety']:
    le = LabelEncoder()
    crop_df[col] = le.fit_transform(crop_df[col])
    label_encoders_crop[col] = le

# === Features & Targets
X_crop = crop_df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
y_crop = crop_df[['Crop', 'Soil_Type', 'Variety']]  # ✅ Now includes Crop

# === Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# === Voting Classifier
voting = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
], voting='soft')

# === Train MultiOutput Model
multi_model = MultiOutputClassifier(voting)
multi_model.fit(X_train, y_train)

# === Save Model & Encoders
joblib.dump(multi_model, "ensemble_crop_model.pkl")
joblib.dump(label_encoders_crop, "crop_label_encoders.pkl")

# === Accuracy (Optional)
print("✅ Crop Model Accuracy:", multi_model.score(X_test, y_test))
