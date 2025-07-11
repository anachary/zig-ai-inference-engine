#!/usr/bin/env python3
"""
Real LLM Raspberry Pi Simulator
Uses actual TinyLLM models for inference instead of simulated responses
"""

import asyncio
import json
import logging
import os
import sys
import time
import yaml
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import psutil

# Add LLM inference path
sys.path.append(os.path.join(os.path.dirname(__file__), '../llm'))

try:
    from tinyllm_inference import TinyLLMInference
    LLM_AVAILABLE = True
except ImportError:
    print("⚠️ TinyLLM not available, falling back to simulated responses")
    LLM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE_NAME = os.getenv("DEVICE_NAME", "real-llm-pi")
DEVICE_PORT = int(os.getenv("DEVICE_PORT", "8081"))
MODEL_NAME = os.getenv("MODEL_NAME", "qwen1.5-0.5b")
USE_REAL_LLM = os.getenv("USE_REAL_LLM", "true").lower() == "true"

class InferenceRequest(BaseModel):
    query: str
    timestamp: Optional[str] = None
    device_id: Optional[str] = None
    priority: str = "normal"
    max_tokens: int = 100
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    result: str
    processing_time_ms: float
    device_name: str
    timestamp: str
    model_used: str
    resource_usage: Dict[str, Any]
    tokens_generated: Optional[int] = None
    inference_type: str = "real_llm"

class RealLLMPiSimulator:
    """Raspberry Pi simulator with real LLM inference"""
    
    def __init__(self, device_name: str, model_name: str = "qwen1.5-0.5b"):
        self.device_name = device_name
        self.model_name = model_name
        self.start_time = time.time()
        self.requests_processed = 0
        self.llm = None
        self.model_loaded = False
        self.device_config = self._load_device_config()
        self.inference_history = []
        
        # Initialize LLM if available
        if LLM_AVAILABLE and USE_REAL_LLM:
            self._initialize_llm()
        else:
            logger.warning("Using simulated responses (LLM not available or disabled)")
    
    def _load_device_config(self) -> Dict[str, Any]:
        """Load device configuration"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "../../config/pi-devices.yaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                
            for device in config.get("devices", []):
                if device["name"] == self.device_name:
                    return device
                    
            return {
                "name": self.device_name,
                "memory_mb": 4096,
                "cpu_cores": 4,
                "use_case": "general"
            }
        except Exception as e:
            logger.warning(f"Could not load device config: {e}")
            return {"name": self.device_name, "memory_mb": 4096, "cpu_cores": 4}
    
    def _initialize_llm(self):
        """Initialize the LLM model"""
        try:
            logger.info(f"Initializing LLM: {self.model_name}")
            self.llm = TinyLLMInference(self.model_name)
            
            # Load model in background to avoid blocking startup
            asyncio.create_task(self._load_model_async())
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
    
    async def _load_model_async(self):
        """Load model asynchronously"""
        try:
            logger.info("Loading LLM model...")
            self.model_loaded = await asyncio.get_event_loop().run_in_executor(
                None, self.llm.load_model
            )
            if self.model_loaded:
                logger.info("✅ LLM model loaded successfully")
            else:
                logger.error("❌ Failed to load LLM model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get real system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Simulate Pi-specific metrics
            temperature = 45.0 + (cpu_percent / 10) + (memory.percent / 20)
            
            return {
                "cpu_usage_percent": round(cpu_percent, 1),
                "memory_usage_percent": round(memory.percent, 1),
                "memory_available_mb": round(memory.available / 1024 / 1024),
                "disk_usage_percent": round(disk.percent, 1),
                "temperature_celsius": round(temperature, 1),
                "load_average": round(cpu_percent / 25, 2)
            }
        except Exception as e:
            logger.warning(f"Could not get system metrics: {e}")
            return {
                "cpu_usage_percent": 30.0,
                "memory_usage_percent": 60.0,
                "memory_available_mb": 2048,
                "disk_usage_percent": 45.0,
                "temperature_celsius": 50.0,
                "load_average": 1.2
            }
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate fallback response when LLM is not available"""
        query_lower = query.lower()
        device_type = self.device_config.get("use_case", "general")
        
        if device_type == "smart-home":
            if "light" in query_lower:
                return "🏠 Smart lighting system activated. Lights adjusted as requested."
            elif "weather" in query_lower:
                return "🌤️ Current weather: 72°F, partly cloudy. Perfect day for IoT demos!"
            elif "timer" in query_lower:
                return "⏰ Timer set successfully. I'll notify you when it's time."
            elif "music" in query_lower:
                return "🎵 Playing your favorite IoT development playlist."
        
        elif device_type == "industrial-iot":
            if "temperature" in query_lower:
                return "🏭 Temperature analysis: Reading within normal operational range."
            elif "vibration" in query_lower:
                return "📊 Vibration analysis: Pattern indicates normal operation."
            elif "pressure" in query_lower:
                return "⚡ Pressure evaluation: Current reading acceptable."
        
        elif device_type == "retail-edge":
            if "product" in query_lower or "find" in query_lower:
                return "🛒 Product location: Check aisle 7 for organic items."
            elif "discount" in query_lower:
                return "💰 Current promotions: 20% off electronics today!"
        
        return f"🤖 Processed '{query}' using fallback response system on {self.device_name}."
    
    async def process_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request with real or fallback LLM"""
        start_time = time.time()
        
        try:
            # Use real LLM if available and loaded
            if self.llm and self.model_loaded and LLM_AVAILABLE:
                logger.info(f"Processing with real LLM: {request.query[:50]}...")
                
                # Run LLM inference in executor to avoid blocking
                llm_result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.llm.generate_response, 
                    request.query, 
                    request.max_tokens
                )
                
                result = llm_result["response"]
                model_used = llm_result.get("model_used", self.model_name)
                tokens_generated = llm_result.get("tokens_generated", 0)
                inference_type = "real_llm"
                
                # Add contextual prefix based on device type
                device_type = self.device_config.get("use_case", "general")
                if device_type == "smart-home" and not result.startswith("🏠"):
                    result = f"🏠 {result}"
                elif device_type == "industrial-iot" and not result.startswith("🏭"):
                    result = f"🏭 {result}"
                elif device_type == "retail-edge" and not result.startswith("🛒"):
                    result = f"🛒 {result}"
                
            else:
                # Fallback to simulated responses
                logger.info(f"Processing with fallback: {request.query[:50]}...")
                result = self._generate_fallback_response(request.query)
                model_used = "fallback-simulator"
                tokens_generated = None
                inference_type = "simulated"
                
                # Simulate processing time
                await asyncio.sleep(1.0 + len(request.query.split()) * 0.1)
            
            # Calculate actual processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update counters
            self.requests_processed += 1
            
            # Get system metrics
            resource_usage = self._get_system_metrics()
            
            # Create response
            response = InferenceResponse(
                result=result,
                processing_time_ms=round(processing_time, 2),
                device_name=self.device_name,
                timestamp=datetime.now().isoformat(),
                model_used=model_used,
                resource_usage=resource_usage,
                tokens_generated=tokens_generated,
                inference_type=inference_type
            )
            
            # Store in history
            self.inference_history.append({
                "timestamp": response.timestamp,
                "query": request.query,
                "response": result[:100] + "..." if len(result) > 100 else result,
                "processing_time_ms": response.processing_time_ms,
                "model_used": response.model_used,
                "inference_type": inference_type
            })
            
            # Keep only last 50 requests
            if len(self.inference_history) > 50:
                self.inference_history = self.inference_history[-50:]
            
            logger.info(f"Inference completed in {processing_time:.2f}ms using {inference_type}")
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Inference failed: {e}")
            
            # Return error response
            return InferenceResponse(
                result=f"Error processing request: {str(e)}",
                processing_time_ms=round(processing_time, 2),
                device_name=self.device_name,
                timestamp=datetime.now().isoformat(),
                model_used="error",
                resource_usage=self._get_system_metrics(),
                inference_type="error"
            )
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get comprehensive device status"""
        uptime = time.time() - self.start_time
        resource_usage = self._get_system_metrics()
        
        last_inference = None
        if self.inference_history:
            last_inference = self.inference_history[-1]["timestamp"]
        
        llm_status = "not_available"
        if LLM_AVAILABLE and USE_REAL_LLM:
            if self.model_loaded:
                llm_status = "loaded"
            elif self.llm:
                llm_status = "loading"
            else:
                llm_status = "failed"
        
        return {
            "device_name": self.device_name,
            "status": "online",
            "uptime_seconds": round(uptime, 2),
            "requests_processed": self.requests_processed,
            "model_name": self.model_name,
            "llm_status": llm_status,
            "model_loaded": self.model_loaded,
            "use_real_llm": USE_REAL_LLM,
            "last_inference_time": last_inference,
            **resource_usage
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        if self.llm and self.model_loaded:
            return self.llm.get_model_info()
        else:
            return {
                "model_name": self.model_name,
                "status": "not_loaded" if LLM_AVAILABLE else "not_available",
                "fallback_mode": True
            }

# Initialize simulator
simulator = RealLLMPiSimulator(DEVICE_NAME, MODEL_NAME)

# FastAPI application
app = FastAPI(
    title=f"Real LLM Pi Simulator - {DEVICE_NAME}",
    description="Raspberry Pi simulator with real TinyLLM inference",
    version="2.0.0"
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main dashboard"""
    status = simulator.get_device_status()
    model_info = simulator.get_model_info()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{DEVICE_NAME} - Real LLM Pi Simulator</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .header {{ text-align: center; color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
            .status {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .card {{ background: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 4px solid #007acc; }}
            .metric {{ margin: 5px 0; }}
            .chat {{ margin: 20px 0; }}
            .chat input {{ width: 70%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            .chat button {{ padding: 10px 20px; background: #007acc; color: white; border: none; border-radius: 5px; cursor: pointer; }}
            .response {{ background: #e8f4fd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .llm-status {{ color: {'green' if status['llm_status'] == 'loaded' else 'orange' if status['llm_status'] == 'loading' else 'red'}; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🍓 {DEVICE_NAME}</h1>
                <p>Real LLM Raspberry Pi Simulator</p>
            </div>
            
            <div class="status">
                <div class="card">
                    <h3>Device Status</h3>
                    <div class="metric">Status: <strong>{status['status']}</strong></div>
                    <div class="metric">Uptime: <strong>{status['uptime_seconds']:.1f}s</strong></div>
                    <div class="metric">Requests: <strong>{status['requests_processed']}</strong></div>
                    <div class="metric">CPU: <strong>{status['cpu_usage_percent']:.1f}%</strong></div>
                    <div class="metric">Memory: <strong>{status['memory_usage_percent']:.1f}%</strong></div>
                    <div class="metric">Temperature: <strong>{status['temperature_celsius']:.1f}°C</strong></div>
                </div>
                
                <div class="card">
                    <h3>LLM Model</h3>
                    <div class="metric">Model: <strong>{model_info.get('model_name', 'Unknown')}</strong></div>
                    <div class="metric">Status: <strong class="llm-status">{status['llm_status']}</strong></div>
                    <div class="metric">Type: <strong>{'Real LLM' if status['use_real_llm'] else 'Simulated'}</strong></div>
                    <div class="metric">Parameters: <strong>{model_info.get('parameters', 'Unknown')}</strong></div>
                    <div class="metric">Device: <strong>{model_info.get('device', 'Unknown')}</strong></div>
                </div>
            </div>
            
            <div class="chat">
                <h3>🤖 Test AI Inference</h3>
                <input type="text" id="queryInput" placeholder="Enter your query..." value="Turn on the living room lights">
                <button onclick="sendQuery()">Send</button>
                <div id="response" class="response" style="display:none;"></div>
            </div>
            
            <div class="card">
                <h3>📊 API Endpoints</h3>
                <div class="metric"><a href="/health">Health Check</a></div>
                <div class="metric"><a href="/api/status">Device Status</a></div>
                <div class="metric"><a href="/api/metrics">Performance Metrics</a></div>
                <div class="metric"><a href="/api/model-info">Model Information</a></div>
                <div class="metric"><a href="/docs">API Documentation</a></div>
            </div>
        </div>
        
        <script>
            async function sendQuery() {{
                const query = document.getElementById('queryInput').value;
                const responseDiv = document.getElementById('response');
                
                responseDiv.style.display = 'block';
                responseDiv.innerHTML = '🤔 Processing...';
                
                try {{
                    const response = await fetch('/api/inference', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ query: query }})
                    }});
                    
                    const result = await response.json();
                    responseDiv.innerHTML = `
                        <strong>🤖 Response:</strong> ${{result.result}}<br>
                        <small>⏱️ Time: ${{result.processing_time_ms}}ms | 
                        🧠 Model: ${{result.model_used}} | 
                        🔧 Type: ${{result.inference_type}}</small>
                    `;
                }} catch (error) {{
                    responseDiv.innerHTML = `<strong>❌ Error:</strong> ${{error.message}}`;
                }}
            }}
            
            // Auto-refresh status every 30 seconds
            setInterval(() => location.reload(), 30000);
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE_NAME,
        "timestamp": datetime.now().isoformat(),
        "llm_available": LLM_AVAILABLE,
        "model_loaded": simulator.model_loaded
    }

@app.get("/api/status")
async def get_status():
    return simulator.get_device_status()

@app.get("/api/model-info")
async def get_model_info():
    return simulator.get_model_info()

@app.post("/api/inference")
async def process_inference(request: InferenceRequest):
    try:
        response = await simulator.process_inference(request)
        return response
    except Exception as e:
        logger.error(f"Inference endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    status = simulator.get_device_status()
    
    # Calculate performance metrics
    recent_requests = simulator.inference_history[-10:] if simulator.inference_history else []
    avg_response_time = 0
    if recent_requests:
        avg_response_time = sum(r["processing_time_ms"] for r in recent_requests) / len(recent_requests)
    
    return {
        "device_name": simulator.device_name,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": status["uptime_seconds"],
        "requests_processed": status["requests_processed"],
        "avg_response_time_ms": round(avg_response_time, 2),
        "cpu_usage_percent": status["cpu_usage_percent"],
        "memory_usage_percent": status["memory_usage_percent"],
        "temperature_celsius": status["temperature_celsius"],
        "model_loaded": status["model_loaded"],
        "llm_status": status["llm_status"],
        "inference_history": recent_requests[-5:]  # Last 5 requests
    }

@app.get("/api/history")
async def get_inference_history():
    return {
        "device_name": simulator.device_name,
        "total_requests": simulator.requests_processed,
        "history": simulator.inference_history[-20:]  # Last 20 requests
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {DEVICE_NAME} with real LLM support")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"LLM Available: {LLM_AVAILABLE}")
    logger.info(f"Use Real LLM: {USE_REAL_LLM}")
    
    uvicorn.run(app, host="0.0.0.0", port=DEVICE_PORT)
