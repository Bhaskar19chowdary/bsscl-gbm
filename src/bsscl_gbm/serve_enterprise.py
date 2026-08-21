import os
import sys
import json
import time
import redis.asyncio as redis
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Import BSSCL-GBM
sys.path.insert(0, '/app/src')
from bsscl_gbm.estimator_v1_2_0 import HybridHistGBMNumbaV1_2_0

app = FastAPI(title="BSSCL-GBM Enterprise API", version="1.2.0")

# Security
API_KEY = "enterprise-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=401, detail="Invalid API Key")

# Global state
model_bsscl = None
model_lgb = None
model_xgb = None
model_cb = None
redis_client = None
models = {}

# Prometheus Metrics
REQUESTS_TOTAL = Counter('requests_total', 'Total prediction requests', ['model_name'])
ERRORS_TOTAL = Counter('prediction_errors_total', 'Total failed requests', ['model_name'])
LATENCY = Histogram('request_latency_seconds', 'Request latency', ['model_name'])

@app.on_event("startup")
async def startup_event():
    global model_bsscl, model_lgb, model_xgb, model_cb, redis_client, models
    
    # Connect to Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    
    # Load/Train Models
    print("Loading models...")
    from sklearn.datasets import make_classification
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    
    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    
    model_bsscl = HybridHistGBMNumbaV1_2_0(n_estimators=100, learning_rate=0.1, random_state=42, verbose=False)
    model_bsscl.fit(X, y)
    
    model_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model_lgb.fit(X, y)
    
    model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X, y)
    
    model_cb = CatBoostClassifier(iterations=100, learning_rate=0.1, random_state=42, verbose=0)
    model_cb.fit(X, y)

    models = {
        "bsscl": model_bsscl,
        "lightgbm": model_lgb,
        "xgboost": model_xgb,
        "catboost": model_cb
    }
    
    print("All models loaded successfully.")

@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()

@app.get("/health")
async def health_check():
    try:
        await redis_client.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "disconnected"
        
    return {
        "status": "healthy" if model_bsscl is not None and redis_status == "ok" else "degraded",
        "redis": redis_status
    }

class PredictionRequest(BaseModel):
    user_ids: List[str]

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

async def _fetch_and_prepare(user_ids: List[str]):
    if not user_ids:
        return [], []
        
    redis_keys = [f"user:{uid}:features" for uid in user_ids]
    raw_features = await redis_client.mget(redis_keys)
    
    valid_indices = []
    X_list = []
    
    for i, raw_str in enumerate(raw_features):
        if raw_str:
            features = json.loads(raw_str)
            X_list.append(features)
            valid_indices.append(i)
            
    if not X_list:
        raise HTTPException(status_code=404, detail="No features found for any provided user_ids.")
        
    X_arr = np.array(X_list, dtype=np.float64)
    return X_arr, valid_indices

@app.post("/v1.2/predict/{model_name}")
async def predict(model_name: str, payload: PredictionRequest, api_key: str = Security(get_api_key)):
    start_time = time.perf_counter()
    REQUESTS_TOTAL.labels(model_name=model_name).inc()
    
    if model_name not in models:
        ERRORS_TOTAL.labels(model_name=model_name).inc()
        raise HTTPException(status_code=404, detail="Model not found")
        
    model = models[model_name]
    
    try:
        X_arr, valid_indices = await _fetch_and_prepare(payload.user_ids)
        if not valid_indices: 
            return {"predictions": []}
            
        preds = model.predict_proba(X_arr)[:, 1].tolist()
        
        results = [None] * len(payload.user_ids)
        for idx, pred in zip(valid_indices, preds): 
            results[idx] = pred
            
        LATENCY.labels(model_name=model_name).observe(time.perf_counter() - start_time)
        return {"predictions": results}
    except HTTPException:
        raise  # Re-raise HTTP exceptions like 404 directly without logging them as 500s
    except Exception as e:
        ERRORS_TOTAL.labels(model_name=model_name).inc()
        raise HTTPException(status_code=500, detail=str(e))
