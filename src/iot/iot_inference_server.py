#!/usr/bin/env python3
"""
IoT Inference Server for Zig AI Platform
Optimized for edge devices with limited resources
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psutil
import yaml

from iot_engine import IoTInferenceEngine
from iot_telemetry import IoTTelemetry
from iot_monitor import IoTMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request/Response models
class InferenceRequest(BaseModel):
    text: Optional[str] = None
    task: str = "classification"
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

class InferenceResponse(BaseModel):
    result: Any
    confidence: Optional[float] = None
    latency_ms: float
    model_version: str
    device_id: str

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    memory_usage_percent: float
    cpu_usage_percent: float
    temperature_celsius: Optional[float] = None
    model_loaded: bool
    inference_count: int

# Global variables
app = FastAPI(
    title="Zig AI IoT Inference Server",
    description="Optimized AI inference for edge devices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
inference_engine: Optional[IoTInferenceEngine] = None
telemetry: Optional[IoTTelemetry] = None
monitor: Optional[IoTMonitor] = None
start_time = time.time()
inference_count = 0

def load_config() -> Dict[str, Any]:
    """Load configuration from file or environment variables"""
    config_path = Path("config/iot-default.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with environment variables
    config.update({
        'device': {
            'type': os.getenv('ZIG_AI_DEVICE_TYPE', 'unknown'),
            'max_memory_mb': int(os.getenv('ZIG_AI_MAX_MEMORY_MB', '512')),
            'threads': int(os.getenv('ZIG_AI_THREADS', '4')),
            'optimization_level': int(os.getenv('ZIG_AI_OPTIMIZATION_LEVEL', '3')),
            'enable_gpu': os.getenv('ZIG_AI_ENABLE_GPU', 'false').lower() == 'true'
        },
        'model': {
            'path': os.getenv('ZIG_AI_MODEL_PATH', '/app/models/current_model.onnx'),
            'cache_size_mb': int(os.getenv('ZIG_AI_CACHE_SIZE_MB', '128'))
        },
        'telemetry': {
            'enabled': os.getenv('ZIG_AI_TELEMETRY_ENABLED', 'false').lower() == 'true',
            'endpoint': os.getenv('ZIG_AI_TELEMETRY_ENDPOINT', ''),
            'interval_seconds': int(os.getenv('ZIG_AI_TELEMETRY_INTERVAL', '60'))
        }
    })
    
    return config

def get_device_temperature() -> Optional[float]:
    """Get device temperature (Raspberry Pi specific)"""
    try:
        # Try Raspberry Pi thermal zone
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read()) / 1000.0
            return temp
    except:
        try:
            # Try vcgencmd for Raspberry Pi
            import subprocess
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                temp = float(temp_str.split('=')[1].replace("'C", ""))
                return temp
        except:
            pass
    return None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global inference_engine, telemetry, monitor
    
    logger.info("🚀 Starting Zig AI IoT Inference Server")
    
    # Load configuration
    config = load_config()
    logger.info(f"📋 Configuration loaded: {config['device']['type']} device")
    
    # Initialize inference engine
    try:
        inference_engine = IoTInferenceEngine(config)
        await inference_engine.initialize()
        logger.info("✅ Inference engine initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize inference engine: {e}")
        raise
    
    # Initialize telemetry (if enabled)
    if config['telemetry']['enabled']:
        try:
            telemetry = IoTTelemetry(config)
            await telemetry.start()
            logger.info("📊 Telemetry enabled")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize telemetry: {e}")
    
    # Initialize monitoring
    try:
        monitor = IoTMonitor(config)
        await monitor.start()
        logger.info("🔍 Monitoring started")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize monitoring: {e}")
    
    logger.info("🎉 IoT Inference Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global inference_engine, telemetry, monitor
    
    logger.info("🛑 Shutting down Zig AI IoT Inference Server")
    
    if monitor:
        await monitor.stop()
    
    if telemetry:
        await telemetry.stop()
    
    if inference_engine:
        await inference_engine.cleanup()
    
    logger.info("👋 Shutdown complete")

@app.post("/v1/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Perform inference on input data"""
    global inference_count
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    start_time = time.time()
    
    try:
        # Prepare input data
        input_data = {
            "text": request.text,
            "task": request.task,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        # Run inference
        result = await inference_engine.infer(input_data)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Update counters
        inference_count += 1
        
        # Log performance metrics
        if inference_count % 100 == 0:
            logger.info(f"📊 Processed {inference_count} inferences, avg latency: {latency_ms:.1f}ms")
        
        return InferenceResponse(
            result=result['output'],
            confidence=result.get('confidence'),
            latency_ms=latency_ms,
            model_version=inference_engine.model_version,
            device_id=inference_engine.device_id
        )
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/v1/inference/image")
async def inference_image(file: UploadFile = File(...), task: str = "classification"):
    """Perform inference on image data"""
    global inference_count
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    start_time = time.time()
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Prepare input data
        input_data = {
            "image": image_data,
            "task": task,
            "filename": file.filename
        }
        
        # Run inference
        result = await inference_engine.infer(input_data)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Update counters
        inference_count += 1
        
        return InferenceResponse(
            result=result['output'],
            confidence=result.get('confidence'),
            latency_ms=latency_ms,
            model_version=inference_engine.model_version,
            device_id=inference_engine.device_id
        )
        
    except Exception as e:
        logger.error(f"❌ Image inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image inference failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=1)
    temperature = get_device_temperature()
    
    return HealthResponse(
        status="healthy" if inference_engine else "unhealthy",
        uptime_seconds=uptime,
        memory_usage_percent=memory_usage,
        cpu_usage_percent=cpu_usage,
        temperature_celsius=temperature,
        model_loaded=inference_engine is not None,
        inference_count=inference_count
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    uptime = time.time() - start_time
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    temperature = get_device_temperature()
    
    metrics_text = f"""# HELP zig_ai_uptime_seconds Total uptime in seconds
# TYPE zig_ai_uptime_seconds counter
zig_ai_uptime_seconds {uptime}

# HELP zig_ai_inference_total Total number of inferences
# TYPE zig_ai_inference_total counter
zig_ai_inference_total {inference_count}

# HELP zig_ai_memory_usage_percent Memory usage percentage
# TYPE zig_ai_memory_usage_percent gauge
zig_ai_memory_usage_percent {memory_usage}

# HELP zig_ai_cpu_usage_percent CPU usage percentage
# TYPE zig_ai_cpu_usage_percent gauge
zig_ai_cpu_usage_percent {cpu_usage}
"""
    
    if temperature is not None:
        metrics_text += f"""
# HELP zig_ai_temperature_celsius Device temperature in Celsius
# TYPE zig_ai_temperature_celsius gauge
zig_ai_temperature_celsius {temperature}
"""
    
    return metrics_text

@app.get("/info")
async def device_info():
    """Get device information"""
    import platform
    
    return {
        "device_type": os.getenv('ZIG_AI_DEVICE_TYPE', 'unknown'),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "model_loaded": inference_engine is not None,
        "model_version": inference_engine.model_version if inference_engine else None,
        "uptime_seconds": time.time() - start_time,
        "inference_count": inference_count
    }

if __name__ == "__main__":
    # Get configuration
    host = os.getenv("ZIG_AI_HOST", "0.0.0.0")
    port = int(os.getenv("ZIG_AI_PORT", "8080"))
    workers = int(os.getenv("ZIG_AI_WORKERS", "1"))
    log_level = os.getenv("ZIG_AI_LOG_LEVEL", "info")
    
    # Run the server
    uvicorn.run(
        "iot_inference_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True
    )
