from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
from datetime import datetime
import sys
import os

# Добавляем текущую папку в PYTHONPATH для Docker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_handler import model_handler

# Создаем приложение
app = FastAPI(
    title="Recommendation System API",
    description="API для системы рекомендаций на основе CatBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настраиваем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных
class RecommendationRequest(BaseModel):
    user_id: int
    limit: Optional[int] = 10

class RecommendationItem(BaseModel):
    item_id: int
    score: float
    user_id: int

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    count: int
    model_version: str = "catboost-v1.0"
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    tree_count: Optional[int] = None

# Эндпоинты
@app.get("/")
async def root():
    return {
        "service": "Recommendation System",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health", 
            "recommendations_POST": "/recommend",
            "recommendations_GET": "/recommend/{user_id}",
            "model_info": "/model-info"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    status = "healthy" if model_handler.model else "unhealthy"
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        model_loaded=bool(model_handler.model),
        tree_count=model_handler.model.tree_count_ if model_handler.model else None
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations_post(request: RecommendationRequest):
    start_time = time.time()
    
    if not model_handler.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        recommendations = model_handler.get_recommendations(
            user_id=request.user_id,
            n_recommendations=request.limit
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            count=len(recommendations),
            processing_time_ms=processing_time_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/{user_id}")
async def get_recommendations_get(user_id: int, limit: int = 10):
    start_time = time.time()
    
    if not model_handler.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        recommendations = model_handler.get_recommendations(
            user_id=user_id,
            n_recommendations=limit
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "model_version": "catboost-v1.0",
            "processing_time_ms": processing_time_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    if not model_handler.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "CatBoostRanker",
        "tree_count": model_handler.model.tree_count_,
        "feature_count": 20,  # Известно из предыдущего вывода
        "categorical_features": [0, 1, 2, 3, 4, 5],
        "loaded_from": "../models/catboost_ranker.bin"
    }

# Запуск сервера
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )