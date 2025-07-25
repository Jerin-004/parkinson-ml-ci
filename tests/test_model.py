import joblib
import pandas as pd

def test_model_prediction():
    df = pd.read_csv("data/parkinson_disease.csv")

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    if 'gender' in df.columns and df['gender'].dtype == object:
        df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    X = df.drop("class", axis=1)

    model = joblib.load("model.pkl")
    preds = model.predict(X)

    assert len(preds) == len(X)
