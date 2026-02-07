import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/face_data.csv", header=None)

X = df.iloc[:, :-1]   
y = df.iloc[:, -1]    

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

joblib.dump(model, "face_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("Model saved successfully")
