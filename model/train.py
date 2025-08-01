import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("data/parkinson_disease.csv")

print(df.columns)
# Drop non-numeric or non-feature columns if needed (like 'id' or 'gender')
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# One-hot encode 'gender' if it's categorical
if 'gender' in df.columns and df['gender'].dtype == object:
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Split features and label
X = df.drop("class", axis=1)
y = df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
