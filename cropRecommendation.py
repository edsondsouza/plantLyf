import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv("Crop_recommendation.csv")
scaler = MinMaxScaler()

# Scale the input features
x_scaled = scaler.fit_transform(data.iloc[:, 0:-1])
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_test)
print(list(y_pred)[10])
print(list(y_test)[10])
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model and the scaler
with open("rf_model.pkl", "wb") as file:
    pickle.dump(rf_classifier, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


