import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'crop_pred.xlsx'
df = pd.read_excel(file_path)

# Select features and target
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph']
target = 'label'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Prompt user for new soil sample values
N = float(input("Enter Nitrogen (N) content: "))
P = float(input("Enter Phosphorus (P) content: "))
K = float(input("Enter Potassium (K) content: "))
temperature = float(input("Enter temperature: "))
humidity = float(input("Enter humidity: "))
ph = float(input("Enter pH level: "))

# Example new soil sample from user input
new_sample = [[N, P, K, temperature, humidity, ph]]
predicted_crop = rf.predict(new_sample)
print('Predicted Crop:', predicted_crop)
