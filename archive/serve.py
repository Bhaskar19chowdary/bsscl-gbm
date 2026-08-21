import os
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model

# Ensure BSSCL-GBM is in path or installed

app = FastAPI(title="BSSCL-GBM Inference API", version="1.2.0")

# Global model instance
model = None
feature_names = []
ModelInputSchema = None

@app.on_event("startup")
async def load_model():
    global model, feature_names, ModelInputSchema
    
    model_path = os.getenv("MODEL_PATH", "model.pkl")
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Model file {model_path} not found. API will not be able to serve predictions until a model is loaded.")
        return
        
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        # Fallback if feature names aren't saved
        n_features = len(model.feature_importances_)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
    # Dynamically create Pydantic schema based on feature names
    fields = {name: (float, ...) for name in feature_names}
    ModelInputSchema = create_model('ModelInputSchema', **fields)
    print("✅ Model loaded successfully.")

class PredictResponse(BaseModel):
    predictions: list[Any]
    probabilities: list[list[float]] = None
    
class ExplainResponse(BaseModel):
    expected_value: float
    contributions: dict[str, float]

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
async def predict(payload: list[dict[str, float]]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    # Validate and extract features in correct order
    try:
        X = np.zeros((len(payload), len(feature_names)), dtype=np.float64)
        for i, row in enumerate(payload):
            for j, fname in enumerate(feature_names):
                X[i, j] = row[fname]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in payload: {e!s}")
        
    # Predict
    preds = model.predict(X).tolist()
    
    res = {"predictions": preds}
    if hasattr(model, "predict_proba"):
        res["probabilities"] = model.predict_proba(X).tolist()
        
    return res

@app.post("/explain", response_model=list[ExplainResponse])
async def explain(payload: list[dict[str, float]]):
    """
    Returns exact local feature contributions using ultra-fast Numba Native TreeSHAP.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    if not hasattr(model, "predict_contributions"):
        raise HTTPException(status_code=501, detail="Model does not support predict_contributions.")
        
    try:
        X = np.zeros((len(payload), len(feature_names)), dtype=np.float64)
        for i, row in enumerate(payload):
            for j, fname in enumerate(feature_names):
                X[i, j] = row[fname]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in payload: {e!s}")
        
    # Get contributions
    contribs = model.predict_contributions(X)
    
    # Handle multi-class vs binary
    if isinstance(contribs, list):
        # Multi-class: return for predicted class for simplicity in API, or return all.
        # For this standard endpoint, we'll return the positive class or argmax class.
        preds = model.predict(X)
        class_idx = [list(model.classes_).index(p) for p in preds]
        
        responses = []
        for i, c_idx in enumerate(class_idx):
            c_mat = contribs[c_idx][i]
            base_val = c_mat[-1]
            feat_contribs = {fname: float(c_mat[j]) for j, fname in enumerate(feature_names)}
            responses.append({"expected_value": float(base_val), "contributions": feat_contribs})
        return responses
    else:
        # Binary / Regression
        responses = []
        for i in range(len(X)):
            c_mat = contribs[i]
            base_val = c_mat[-1]
            feat_contribs = {fname: float(c_mat[j]) for j, fname in enumerate(feature_names)}
            responses.append({"expected_value": float(base_val), "contributions": feat_contribs})
        return responses
