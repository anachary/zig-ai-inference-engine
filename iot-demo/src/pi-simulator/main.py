#!/usr/bin/env python3
"""
Raspberry Pi Simulator for Zig AI Platform IoT Demo
Simulates a Raspberry Pi device running lightweight LLM inference
"""

import asyncio
import json
import logging
import os
import time
import yaml
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psutil
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration from environment
DEVICE_NAME = os.getenv("DEVICE_NAME", "generic-pi")
DEVICE_PORT = int(os.getenv("DEVICE_PORT", "8081"))
COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://localhost:8080")

class InferenceRequest(BaseModel):
    query: str
    timestamp: Optional[str] = None
    device_id: Optional[str] = None
    priority: str = "normal"
    max_tokens: int = 100

class InferenceResponse(BaseModel):
    result: str
    processing_time_ms: float
    device_name: str
    timestamp: str
    model_used: str
    resource_usage: Dict[str, Any]

class DeviceStatus(BaseModel):
    device_name: str
    status: str
    uptime_seconds: float
    cpu_usage_percent: float
    memory_usage_percent: float
    temperature_celsius: float
    model_loaded: Optional[str]
    requests_processed: int
    last_inference_time: Optional[str]

class RaspberryPiSimulator:
    """Simulates a Raspberry Pi device with resource constraints"""
    
    def __init__(self, device_name: str):
        self.device_name = device_name
        self.start_time = time.time()
        self.requests_processed = 0
        self.current_model = None
        self.device_config = self._load_device_config()
        self.model_config = self._load_model_config()
        self.inference_history = []
        
        # Simulate hardware constraints
        self.max_memory_mb = self.device_config.get("memory_mb", 4096)
        self.cpu_cores = self.device_config.get("cpu_cores", 4)
        self.architecture = self.device_config.get("architecture", "arm64")
        
        logger.info(f"Initialized {device_name} simulator with {self.max_memory_mb}MB RAM, {self.cpu_cores} cores")
    
    def _load_device_config(self) -> Dict[str, Any]:
        """Load device configuration from YAML file"""
        try:
            with open("/app/config/pi-devices.yaml", "r") as f:
                config = yaml.safe_load(f)
                
            # Find this device's configuration
            for device in config.get("devices", []):
                if device["name"] == self.device_name:
                    return device
                    
            # Return default configuration if not found
            return {
                "name": self.device_name,
                "memory_mb": 4096,
                "cpu_cores": 4,
                "architecture": "arm64",
                "network": {"type": "wifi", "bandwidth_mbps": 50}
            }
        except Exception as e:
            logger.warning(f"Could not load device config: {e}")
            return {"name": self.device_name, "memory_mb": 4096, "cpu_cores": 4}
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file"""
        try:
            with open("/app/config/models.yaml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load model config: {e}")
            return {"models": []}
    
    def _select_optimal_model(self, use_case: str) -> Dict[str, Any]:
        """Select the best model for the device's use case"""
        device_use_case = self.device_config.get("use_case", "general")
        
        for model in self.model_config.get("models", []):
            if device_use_case in model.get("use_cases", []):
                if model.get("memory_requirement_mb", 0) <= self.max_memory_mb:
                    return model
        
        # Fallback to smallest model
        models = self.model_config.get("models", [])
        if models:
            return min(models, key=lambda m: m.get("memory_requirement_mb", 0))
        
        # Default model if none configured
        return {
            "name": "default-llm",
            "memory_requirement_mb": 1000,
            "inference_time_ms": 2000
        }
    
    def _simulate_resource_usage(self) -> Dict[str, Any]:
        """Simulate realistic resource usage during inference"""
        base_cpu = 30 + random.uniform(-10, 20)  # 20-50% CPU usage
        base_memory = 60 + random.uniform(-15, 25)  # 45-85% memory usage
        base_temp = 45 + random.uniform(-5, 15)  # 40-60°C temperature
        
        # Add load-based variations
        load_factor = min(self.requests_processed / 100, 1.0)
        cpu_usage = min(base_cpu + (load_factor * 20), 95)
        memory_usage = min(base_memory + (load_factor * 15), 90)
        temperature = min(base_temp + (load_factor * 10), 75)
        
        return {
            "cpu_usage_percent": round(cpu_usage, 1),
            "memory_usage_percent": round(memory_usage, 1),
            "temperature_celsius": round(temperature, 1),
            "memory_available_mb": round(self.max_memory_mb * (1 - memory_usage/100)),
            "load_average": round(cpu_usage / 25, 2)
        }
    
    def _simulate_inference(self, query: str, model: Dict[str, Any]) -> tuple[str, float]:
        """Simulate LLM inference with realistic timing and responses"""
        start_time = time.time()
        
        # Simulate processing time based on model and query complexity
        base_time = model.get("inference_time_ms", 2000)
        query_complexity = len(query.split()) / 10  # Rough complexity measure
        network_latency = self.device_config.get("network", {}).get("latency_ms", 20)
        
        # Add realistic variations
        processing_time = base_time + (query_complexity * 100) + network_latency
        processing_time *= random.uniform(0.8, 1.3)  # ±30% variation
        
        # Simulate actual processing delay
        time.sleep(processing_time / 1000)  # Convert to seconds
        
        # Generate contextual response based on device type
        response = self._generate_contextual_response(query)
        
        actual_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return response, actual_time
    
    def _generate_contextual_response(self, query: str) -> str:
        """Generate contextual responses based on device type and query"""
        device_type = self.device_config.get("use_case", "general")
        query_lower = query.lower()
        
        if device_type == "smart-home":
            if "light" in query_lower:
                return "I've adjusted the lighting as requested. The smart home system is responding."
            elif "weather" in query_lower:
                return "Today's weather: 72°F, partly cloudy with light winds. Perfect day!"
            elif "timer" in query_lower:
                return "Timer set successfully. I'll notify you when it's time."
            elif "music" in query_lower:
                return "Playing your relaxing music playlist from the living room speakers."
            else:
                return f"Smart home assistant processed your request: '{query}'. System is ready for next command."
        
        elif device_type == "industrial-iot":
            if "temperature" in query_lower:
                return "Temperature analysis: Reading is within normal operational range. No action required."
            elif "vibration" in query_lower:
                return "Vibration analysis: Pattern indicates normal operation. Recommend monitoring trend."
            elif "pressure" in query_lower:
                return "Pressure evaluation: Current reading is acceptable. Schedule routine maintenance check."
            elif "current" in query_lower or "motor" in query_lower:
                return "Motor current analysis: Operating efficiently. Predictive maintenance suggests service in 3-4 weeks."
            else:
                return f"Industrial AI analyzed sensor data: '{query}'. System status: operational."
        
        elif device_type == "retail-edge":
            if "vegetable" in query_lower or "organic" in query_lower:
                return "Organic vegetables are located in Aisle 7, fresh produce section on your left."
            elif "discount" in query_lower or "sale" in query_lower:
                return "Today's electronics sale: 20% off tablets, 15% off headphones. Check endcap displays!"
            elif "stock" in query_lower:
                return "Let me check our inventory system... Item appears to be in stock. Aisle 3, shelf B."
            elif "hours" in query_lower:
                return "Store hours: Monday-Saturday 8AM-10PM, Sunday 9AM-9PM. We're open now!"
            else:
                return f"Retail assistant helped with: '{query}'. Is there anything else I can help you find?"
        
        else:
            return f"AI assistant processed: '{query}'. Response generated successfully on {self.device_name}."
    
    async def process_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Process an inference request"""
        logger.info(f"Processing inference request: {request.query[:50]}...")
        
        # Select appropriate model
        model = self._select_optimal_model(request.query)
        self.current_model = model["name"]
        
        # Simulate inference
        result, processing_time = self._simulate_inference(request.query, model)
        
        # Update counters
        self.requests_processed += 1
        
        # Get current resource usage
        resource_usage = self._simulate_resource_usage()
        
        # Create response
        response = InferenceResponse(
            result=result,
            processing_time_ms=round(processing_time, 2),
            device_name=self.device_name,
            timestamp=datetime.now().isoformat(),
            model_used=model["name"],
            resource_usage=resource_usage
        )
        
        # Store in history
        self.inference_history.append({
            "timestamp": response.timestamp,
            "query": request.query,
            "processing_time_ms": response.processing_time_ms,
            "model_used": response.model_used
        })
        
        # Keep only last 100 requests in history
        if len(self.inference_history) > 100:
            self.inference_history = self.inference_history[-100:]
        
        logger.info(f"Inference completed in {processing_time:.2f}ms")
        return response
    
    def get_device_status(self) -> DeviceStatus:
        """Get current device status"""
        uptime = time.time() - self.start_time
        resource_usage = self._simulate_resource_usage()
        
        last_inference = None
        if self.inference_history:
            last_inference = self.inference_history[-1]["timestamp"]
        
        return DeviceStatus(
            device_name=self.device_name,
            status="online",
            uptime_seconds=round(uptime, 2),
            cpu_usage_percent=resource_usage["cpu_usage_percent"],
            memory_usage_percent=resource_usage["memory_usage_percent"],
            temperature_celsius=resource_usage["temperature_celsius"],
            model_loaded=self.current_model,
            requests_processed=self.requests_processed,
            last_inference_time=last_inference
        )

# Initialize the simulator
simulator = RaspberryPiSimulator(DEVICE_NAME)

# FastAPI application
app = FastAPI(title=f"Zig AI Platform - {DEVICE_NAME}", version="1.0.0")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "device": DEVICE_NAME, "timestamp": datetime.now().isoformat()}

@app.get("/api/device-info")
async def get_device_info():
    """Get device information"""
    return {
        "device_name": simulator.device_name,
        "device_config": simulator.device_config,
        "model_config": simulator.model_config,
        "architecture": simulator.architecture,
        "max_memory_mb": simulator.max_memory_mb,
        "cpu_cores": simulator.cpu_cores
    }

@app.get("/api/status")
async def get_status():
    """Get current device status"""
    return simulator.get_device_status()

@app.post("/api/inference")
async def process_inference(request: InferenceRequest):
    """Process an inference request"""
    try:
        response = await simulator.process_inference(request)
        return response
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/api/history")
async def get_inference_history():
    """Get inference history"""
    return {
        "device_name": simulator.device_name,
        "total_requests": simulator.requests_processed,
        "history": simulator.inference_history[-20:]  # Last 20 requests
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get device metrics for monitoring"""
    status = simulator.get_device_status()
    
    # Calculate performance metrics
    recent_requests = simulator.inference_history[-10:] if simulator.inference_history else []
    avg_response_time = 0
    if recent_requests:
        avg_response_time = sum(r["processing_time_ms"] for r in recent_requests) / len(recent_requests)
    
    return {
        "device_name": simulator.device_name,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": status.uptime_seconds,
        "requests_processed": status.requests_processed,
        "avg_response_time_ms": round(avg_response_time, 2),
        "cpu_usage_percent": status.cpu_usage_percent,
        "memory_usage_percent": status.memory_usage_percent,
        "temperature_celsius": status.temperature_celsius,
        "model_loaded": status.model_loaded,
        "requests_per_minute": len([r for r in recent_requests if 
                                   (datetime.now() - datetime.fromisoformat(r["timestamp"].replace('Z', '+00:00'))).total_seconds() < 60])
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {DEVICE_NAME} simulator on port {DEVICE_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=DEVICE_PORT)
