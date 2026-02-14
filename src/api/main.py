"""
ConsensusRetNet API - Model 4 Prediction Service
=================================================

FastAPI backend serving the trained RetNet model for real-time 
consensus mechanism prediction in vehicular blockchain networks.

Endpoints:
  GET  /                     - Service info
  GET  /api/v1/health        - Health check
  GET  /api/v1/model/info    - Model architecture info
  POST /api/v1/predict       - Single prediction
  POST /api/v1/predict/batch - Batch prediction
  GET  /api/v1/stats         - Request statistics

Author: NOK KO
Date: 2026-02-05
"""

import time
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import torch

# Add parent paths
SRC_DIR = Path(__file__).parent.parent          # src/
BASE_DIR = SRC_DIR.parent                       # GIT_MODEL4/
sys.path.append(str(SRC_DIR))
sys.path.append(str(SRC_DIR / 'models'))

from models.retnet_consensus import create_model

# ==============================================================================
# Global State
# ==============================================================================

class ModelEngine:
    """Inference engine managing the trained RetNet model"""
    
    def __init__(self):
        self.model = None
        self.feature_stats = None
        self.device = 'cpu'
        self.loaded = False
        self.checkpoint_info = {}
        self.feature_keys = [
            'num_nodes', 'connectivity', 'latency_requirement_sec',
            'throughput_requirement_tps', 'byzantine_tolerance', 'security_priority',
            'energy_budget', 'bandwidth_mbps', 'consistency_requirement',
            'decentralization_requirement', 'network_load', 'attack_risk'
        ]
        self.class_names = ['PoW', 'PoS', 'PBFT', 'DPoS', 'Hybrid']
        self.class_full_names = {
            'PoW': 'Proof of Work',
            'PoS': 'Proof of Stake',
            'PBFT': 'Practical Byzantine Fault Tolerance',
            'DPoS': 'Delegated Proof of Stake',
            'Hybrid': 'Hybrid Consensus',
        }
    
    def load(self):
        """Load model from checkpoint"""
        ckpt_path = BASE_DIR / 'training' / 'checkpoints' / 'best_consensus.pth'
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        checkpoint = torch.load(str(ckpt_path), weights_only=False, map_location=self.device)
        
        self.model = create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        
        self.feature_stats = checkpoint.get('feature_stats', None)
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_accuracy': checkpoint.get('val_acc', 'N/A'),
            'val_loss': checkpoint.get('val_loss', 'N/A'),
            'class_accuracy': checkpoint.get('class_acc', {}),
        }
        self.loaded = True
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model loaded: ConsensusRetNet ({n_params:,} params)")
        print(f"  Val accuracy: {self.checkpoint_info['val_accuracy']:.2f}%")
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_stats is None:
            return x
        mean = self.feature_stats['mean']
        std = self.feature_stats['std']
        mean_t = torch.tensor(mean, dtype=torch.float32) if not isinstance(mean, torch.Tensor) else mean
        std_t = torch.tensor(std, dtype=torch.float32) if not isinstance(std, torch.Tensor) else std
        return (x - mean_t.unsqueeze(0)) / (std_t.unsqueeze(0) + 1e-8)
    
    def predict(self, network_state: Dict) -> Dict:
        """Predict optimal consensus mechanism"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        features = [network_state.get(k, 0.0) for k in self.feature_keys]
        x = torch.tensor([features], dtype=torch.float32)
        x_norm = self.normalize(x)
        
        with torch.no_grad():
            logits = self.model(x_norm)
            probs = torch.softmax(logits, dim=1)[0]
        
        pred_idx = probs.argmax().item()
        predicted_class = self.class_names[pred_idx]
        confidence = probs[pred_idx].item()
        
        return {
            'predicted_consensus': predicted_class,
            'full_name': self.class_full_names[predicted_class],
            'confidence': round(confidence, 6),
            'probabilities': {
                c: round(probs[i].item(), 6) for i, c in enumerate(self.class_names)
            },
            'input_features': {k: features[i] for i, k in enumerate(self.feature_keys)},
        }
    
    def predict_batch(self, states: List[Dict]) -> List[Dict]:
        """Batch prediction"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        features_list = []
        for state in states:
            features = [state.get(k, 0.0) for k in self.feature_keys]
            features_list.append(features)
        
        x = torch.tensor(features_list, dtype=torch.float32)
        x_norm = self.normalize(x)
        
        with torch.no_grad():
            logits = self.model(x_norm)
            probs = torch.softmax(logits, dim=1)
        
        results = []
        for j in range(len(states)):
            pred_idx = probs[j].argmax().item()
            predicted_class = self.class_names[pred_idx]
            results.append({
                'predicted_consensus': predicted_class,
                'full_name': self.class_full_names[predicted_class],
                'confidence': round(probs[j][pred_idx].item(), 6),
                'probabilities': {
                    c: round(probs[j][i].item(), 6) for i, c in enumerate(self.class_names)
                },
            })
        
        return results


# Global engine
engine = ModelEngine()

# Stats
stats = {
    'start_time': time.time(),
    'requests_served': 0,
    'batch_requests': 0,
    'total_predictions': 0,
}

# ==============================================================================
# FastAPI App
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle"""
    print("=" * 60)
    print("  ConsensusRetNet API - Starting...")
    print("=" * 60)
    engine.load()
    print("  API ready at http://localhost:8000")
    print("  Docs at http://localhost:8000/docs")
    print("=" * 60)
    yield
    print("  API shutting down...")

app = FastAPI(
    title="ConsensusRetNet API",
    description="Model 4: RetNet-based consensus mechanism prediction for vehicular blockchain networks",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Request/Response Models
# ==============================================================================

class NetworkState(BaseModel):
    """Input: vehicular network conditions"""
    num_nodes: float = Field(100, description="Number of network nodes [10-1000]")
    connectivity: float = Field(0.8, description="Network connectivity ratio [0.6-0.95]")
    latency_requirement_sec: float = Field(10.0, description="Max tolerable latency in seconds [0.5-60]")
    throughput_requirement_tps: float = Field(500, description="Required throughput TPS [5-10000]")
    byzantine_tolerance: float = Field(0.15, description="Byzantine fault tolerance ratio [0-0.33]")
    security_priority: float = Field(0.8, description="Security importance [0.6-1.0]")
    energy_budget: float = Field(0.5, description="Energy budget (normalized) [0.1-1.0]")
    bandwidth_mbps: float = Field(1000, description="Network bandwidth in Mbps [100-10000]")
    consistency_requirement: float = Field(0.75, description="Consistency strength [0.5-1.0]")
    decentralization_requirement: float = Field(0.7, description="Decentralization need [0.3-1.0]")
    network_load: float = Field(0.5, description="Current network utilization [0.1-0.9]")
    attack_risk: float = Field(0.3, description="Perceived attack risk [0-1.0]")

class PredictRequest(BaseModel):
    network_state: NetworkState

class BatchPredictRequest(BaseModel):
    network_states: List[NetworkState]

class PredictionResponse(BaseModel):
    success: bool
    prediction: Dict
    metadata: Dict

class BatchPredictionResponse(BaseModel):
    success: bool
    predictions: List[Dict]
    metadata: Dict


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/")
async def root():
    return {
        "service": "ConsensusRetNet API",
        "model": "Model 4: Consensus Mechanism Selection",
        "architecture": "RetNet (ICML 2023)",
        "version": "1.0.0",
        "status": "running" if engine.loaded else "loading",
        "endpoints": {
            "health": "/api/v1/health",
            "model_info": "/api/v1/model/info",
            "predict": "/api/v1/predict",
            "predict_batch": "/api/v1/predict/batch",
            "stats": "/api/v1/stats",
            "docs": "/docs",
        }
    }

@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy" if engine.loaded else "unhealthy",
        "model_loaded": engine.loaded,
        "device": engine.device,
        "uptime_seconds": round(time.time() - stats['start_time'], 1),
        "requests_served": stats['requests_served'],
    }

@app.get("/api/v1/model/info")
async def model_info():
    if not engine.loaded:
        raise HTTPException(500, "Model not loaded")
    
    n_params = sum(p.numel() for p in engine.model.parameters())
    return {
        "model_name": "ConsensusRetNet",
        "architecture": "RetNet (Retentive Network - ICML 2023)",
        "task": "Multi-class classification (5 consensus mechanisms)",
        "parameters": n_params,
        "input_features": engine.feature_keys,
        "output_classes": engine.class_names,
        "class_descriptions": engine.class_full_names,
        "d_model": 384,
        "num_layers": 3,
        "num_heads": 3,
        "retention_gammas": [0.9, 0.95, 0.99],
        "checkpoint": engine.checkpoint_info,
    }

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest):
    if not engine.loaded:
        raise HTTPException(500, "Model not loaded")
    
    t0 = time.time()
    state_dict = request.network_state.model_dump()
    prediction = engine.predict(state_dict)
    inference_time = (time.time() - t0) * 1000
    
    stats['requests_served'] += 1
    stats['total_predictions'] += 1
    
    return PredictionResponse(
        success=True,
        prediction=prediction,
        metadata={
            "inference_time_ms": round(inference_time, 2),
            "model": "ConsensusRetNet",
            "device": engine.device,
        }
    )

@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictRequest):
    if not engine.loaded:
        raise HTTPException(500, "Model not loaded")
    
    t0 = time.time()
    states = [s.model_dump() for s in request.network_states]
    predictions = engine.predict_batch(states)
    inference_time = (time.time() - t0) * 1000
    
    stats['requests_served'] += 1
    stats['batch_requests'] += 1
    stats['total_predictions'] += len(predictions)
    
    return BatchPredictionResponse(
        success=True,
        predictions=predictions,
        metadata={
            "batch_size": len(predictions),
            "inference_time_ms": round(inference_time, 2),
            "avg_time_per_sample_ms": round(inference_time / max(len(predictions), 1), 2),
            "model": "ConsensusRetNet",
        }
    )

@app.get("/api/v1/stats")
async def get_stats():
    return {
        "uptime_seconds": round(time.time() - stats['start_time'], 1),
        "requests_served": stats['requests_served'],
        "batch_requests": stats['batch_requests'],
        "total_predictions": stats['total_predictions'],
    }


# ==============================================================================
# Run
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
