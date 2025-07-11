#!/usr/bin/env python3
"""
Simplified Raspberry Pi Simulator for Online Environments
"""

import asyncio
import json
import time
import random
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import psutil
import os

# Configuration
DEVICE_NAME = os.getenv("DEVICE_NAME", "online-pi-simulator")
DEVICE_PORT = int(os.getenv("DEVICE_PORT", "8081"))

class InferenceRequest(BaseModel):
    query: str
    device_id: str = None
    priority: str = "normal"

class InferenceResponse(BaseModel):
    result: str
    processing_time_ms: float
    device_name: str
    timestamp: str
    model_used: str
    resource_usage: dict

class PiSimulator:
    def __init__(self, device_name: str):
        self.device_name = device_name
        self.start_time = time.time()
        self.requests_processed = 0
        
    def get_system_metrics(self):
        """Get real system metrics from the online environment"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage_percent": round(cpu_percent, 1),
                "memory_usage_percent": round(memory.percent, 1),
                "memory_available_mb": round(memory.available / 1024 / 1024),
                "disk_usage_percent": round(disk.percent, 1),
                "temperature_celsius": round(45 + random.uniform(-5, 15), 1)  # Simulated
            }
        except:
            # Fallback if psutil fails
            return {
                "cpu_usage_percent": round(30 + random.uniform(-10, 20), 1),
                "memory_usage_percent": round(60 + random.uniform(-15, 25), 1),
                "memory_available_mb": 2048,
                "disk_usage_percent": 45.0,
                "temperature_celsius": round(50 + random.uniform(-5, 10), 1)
            }
    
    def generate_response(self, query: str) -> str:
        """Generate contextual AI responses"""
        query_lower = query.lower()
        
        # Smart Home responses
        if "light" in query_lower:
            return f"🏠 Smart lighting system activated. Adjusting lights as requested."
        elif "weather" in query_lower:
            return f"🌤️ Current weather: 72°F, partly cloudy. Perfect day for IoT demos!"
        elif "timer" in query_lower:
            return f"⏰ Timer set successfully. I'll notify you when time's up."
        elif "music" in query_lower:
            return f"🎵 Playing your favorite IoT development playlist."
        
        # Industrial responses
        elif "temperature" in query_lower or "sensor" in query_lower:
            return f"🏭 Sensor analysis: Reading within normal parameters. System operational."
        elif "vibration" in query_lower:
            return f"📊 Vibration analysis: Pattern indicates normal operation. Monitoring continues."
        elif "pressure" in query_lower:
            return f"⚡ Pressure evaluation: Current reading acceptable. No action required."
        
        # Retail responses
        elif "store" in query_lower or "product" in query_lower:
            return f"🛒 Store assistant: I can help you find what you're looking for!"
        elif "discount" in query_lower or "sale" in query_lower:
            return f"💰 Current promotions: Check our IoT device sale - 20% off today!"
        
        # General responses
        else:
            responses = [
                f"🤖 AI processed your query: '{query}' successfully on {self.device_name}",
                f"🧠 Neural network analysis complete. Query understood and processed.",
                f"⚡ Edge AI inference completed. Response generated locally on device.",
                f"🔬 Machine learning model processed your request efficiently.",
                f"🚀 Distributed AI system handled your query with optimal performance."
            ]
            return random.choice(responses)
    
    async def process_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request with realistic timing"""
        start_time = time.time()
        
        # Simulate processing time based on query complexity
        base_time = 1.5  # Base processing time in seconds
        complexity_factor = len(request.query.split()) / 10
        processing_time = base_time + complexity_factor + random.uniform(-0.5, 1.0)
        
        # Simulate processing delay
        await asyncio.sleep(processing_time)
        
        # Generate response
        result = self.generate_response(request.query)
        
        # Update counters
        self.requests_processed += 1
        
        # Get system metrics
        resource_usage = self.get_system_metrics()
        
        actual_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return InferenceResponse(
            result=result,
            processing_time_ms=round(actual_time, 2),
            device_name=self.device_name,
            timestamp=datetime.now().isoformat(),
            model_used="edge-optimized-llm-online",
            resource_usage=resource_usage
        )

# Initialize simulator
simulator = PiSimulator(DEVICE_NAME)

# FastAPI app
app = FastAPI(title=f"Online Pi Simulator - {DEVICE_NAME}", version="1.0.0")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {DEVICE_NAME} Online Simulator!",
        "status": "online",
        "environment": "cloud-based",
        "uptime_seconds": round(time.time() - simulator.start_time, 2)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE_NAME,
        "timestamp": datetime.now().isoformat(),
        "environment": "online-simulator"
    }

@app.get("/api/status")
async def get_status():
    uptime = time.time() - simulator.start_time
    metrics = simulator.get_system_metrics()
    
    return {
        "device_name": simulator.device_name,
        "status": "online",
        "uptime_seconds": round(uptime, 2),
        "requests_processed": simulator.requests_processed,
        "cpu_usage_percent": metrics["cpu_usage_percent"],
        "memory_usage_percent": metrics["memory_usage_percent"],
        "temperature_celsius": metrics["temperature_celsius"],
        "environment": "online-simulator"
    }

@app.post("/api/inference")
async def process_inference(request: InferenceRequest):
    try:
        response = await simulator.process_inference(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    uptime = time.time() - simulator.start_time
    metrics = simulator.get_system_metrics()
    
    return {
        "device_name": simulator.device_name,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(uptime, 2),
        "requests_processed": simulator.requests_processed,
        "environment": "online-simulator",
        **metrics
    }

if __name__ == "__main__":
    print(f"🚀 Starting {DEVICE_NAME} Online Simulator on port {DEVICE_PORT}")
    print(f"🌐 Access at: http://localhost:{DEVICE_PORT}")
    print(f"📊 Health check: http://localhost:{DEVICE_PORT}/health")
    print(f"🔧 API docs: http://localhost:{DEVICE_PORT}/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=DEVICE_PORT)
