import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


dataset_path = "crop_data.csv"
data = pd.read_csv(dataset_path)

X = data.drop(columns=['label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def recommend_crop(n, p, k, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame({'N': [n],'P': [p],'K': [k],'temperature': [temperature],
                               'humidity': [humidity],'ph': [ph],'rainfall': [rainfall]})
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
