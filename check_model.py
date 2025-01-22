import pandas as pd
import joblib

model_path = "crop_recommendation_model.joblib"
clf = joblib.load(model_path)
print("Model loaded successfully.")

def recommend_crop(n, p, k, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame({'N': [n], 'P': [p], 'K': [k],
                               'temperature': [temperature], 'humidity': [humidity],
                               'ph': [ph], 'rainfall': [rainfall]})
    prediction = clf.predict(input_data)
    return prediction[0]

n = float(input("Enter Nitrogen content (N): "))
p = float(input("Enter Phosphorus content (P): "))
k = float(input("Enter Potassium content (K): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall (mm): "))

recommended_crop = recommend_crop(n, p, k, temperature, humidity, ph, rainfall)
print(f"Recommended Crop: {recommended_crop}")
