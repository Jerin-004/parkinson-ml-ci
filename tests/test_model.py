import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def test_model_function():
    # Load dataset and model
    print("Loading dataset and model...")
    df = pd.read_csv("data/parkinson_disease.csv")

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    if 'gender' in df.columns and df['gender'].dtype == object:
        df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    X = df.drop("class", axis=1)
    y = df["class"] 

    model = joblib.load("model.pkl")
    preds = model.predict(X)

    assert len(preds) == len(X)


    # Calculate accuracy
    acc = accuracy_score(y, preds)
    print("Accuracy:", acc)

    # Optional: print confusion matrix and classification report
    print("Confusion Matrix:\n", confusion_matrix(y, preds))
    print("Classification Report:\n", classification_report(y, preds))

if __name__ == "__main__":
    test_model_function()
