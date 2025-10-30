import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from flask import Flask, request, jsonify 
import joblib 
data = pd.read_csv("diabetes.csv") 
print(data.head()) 
print(data.isnull().sum()) 
X = data.drop('Outcome', axis=1) 
y = data['Outcome'] 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split( 
X_scaled, y, test_size=0.2, random_state=42 
3 
) 
model = LogisticRegression(max_iter=200) 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
print("Classification Report:\n", classification_report(y_test, y_pred)) 
joblib.dump((scaler, model), 'diabetes_model.pkl') 
app = Flask(__name__) 
scaler, model = joblib.load('diabetes_model.pkl') 
@app.route('/predict', methods=['POST']) 
def predict(): 
data = request.get_json(force=True) 
features = np.array(data['features']).reshape(1, -1) 
scaled_features = scaler.transform(features) 
prediction = model.predict(scaled_features)[0] 
result = "Diabetic" if prediction == 1 else "Non-Diabetic" 
return jsonify({'prediction': result}) 
4 
if __name__ == "__main__": 
app.run(port=5000, debug=False)
