import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np


app = FastAPI()


class ScoringItem(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


with open('logreg.pkl', 'rb') as f:
    model = pickle.load(f)

try:
    imputer = model.named_steps.get('imputer') or model.steps[0][1]
    if not hasattr(imputer, '_fill_dtype'):
        imputer._fill_dtype = imputer._fit_dtype
except Exception:
    pass


@app.post("/")
async def scoring_item(item: ScoringItem):

    df = pd.DataFrame([item.model_dump()])

    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}
