#!/usr/bin/env python3
"""
Quick Test - Raspberry Pi IoT Simulator
Copy and run this file in any Python environment to test the concept
"""

import asyncio
import json
import time
import random
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Simple models
class InferenceRequest(BaseModel):
    query: str
    device_id: str = "test-pi"

class InferenceResponse(BaseModel):
    result: str
    processing_time_ms: float
    device_name: str
    timestamp: str

# Raspberry Pi Simulator
class QuickPiSimulator:
    def __init__(self):
        self.device_name = "quick-test-pi"
        self.start_time = time.time()
        self.requests = 0
        
    def get_metrics(self):
        return {
            "cpu_percent": round(random.uniform(20, 80), 1),
            "memory_percent": round(random.uniform(40, 85), 1),
            "temperature": round(random.uniform(45, 65), 1),
            "uptime": round(time.time() - self.start_time, 1)
        }
    
    def generate_ai_response(self, query: str) -> str:
        """Generate realistic AI responses based on query type"""
        q = query.lower()
        
        if "light" in q or "lamp" in q:
            return "🏠 Smart lighting system activated. Lights adjusted as requested."
        elif "weather" in q:
            return "🌤️ Current weather: 72°F, partly cloudy. Great day for IoT development!"
        elif "temperature" in q:
            return "🌡️ Current temperature: 68°F. All sensors reading normal."
        elif "music" in q or "play" in q:
            return "🎵 Playing your IoT development playlist. Enjoy coding!"
        elif "timer" in q:
            return "⏰ Timer set successfully. I'll notify you when it's time."
        elif "status" in q or "system" in q:
            return "✅ All systems operational. IoT network running smoothly."
        elif "hello" in q or "hi" in q:
            return "👋 Hello! I'm your Raspberry Pi AI assistant. How can I help?"
        elif "sensor" in q:
            return "📊 Sensor data analyzed. All readings within normal parameters."
        elif "security" in q:
            return "🔒 Security system active. All zones secure."
        elif "energy" in q or "power" in q:
            return "⚡ Energy usage optimized. Running efficiently on edge computing."
        else:
            responses = [
                f"🤖 Processed '{query}' using edge AI on Raspberry Pi simulator.",
                f"🧠 Neural network analysis complete. Query understood and processed.",
                f"⚡ Edge inference completed successfully. Response generated locally.",
                f"🔬 Machine learning model processed your request efficiently.",
                f"🚀 Distributed AI system handled your query with optimal performance."
            ]
            return random.choice(responses)
    
    async def process_inference(self, request: InferenceRequest) -> InferenceResponse:
        start_time = time.time()
        
        # Simulate realistic processing time (1-4 seconds)
        processing_delay = 1.0 + len(request.query.split()) * 0.1 + random.uniform(0, 2)
        await asyncio.sleep(processing_delay)
        
        # Generate AI response
        result = self.generate_ai_response(request.query)
        
        # Update counters
        self.requests += 1
        
        actual_time = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            result=result,
            processing_time_ms=round(actual_time, 2),
            device_name=self.device_name,
            timestamp=datetime.now().isoformat()
        )

# Initialize simulator
simulator = QuickPiSimulator()

# FastAPI app
app = FastAPI(title="Quick Raspberry Pi IoT Simulator", version="1.0.0")

@app.get("/")
async def root():
    metrics = simulator.get_metrics()
    return {
        "message": "🍓 Raspberry Pi IoT Simulator Running!",
        "device": simulator.device_name,
        "status": "online",
        "uptime_seconds": metrics["uptime"],
        "requests_processed": simulator.requests,
        "demo_queries": [
            "Turn on the lights",
            "What's the weather?",
            "Check system status",
            "Set a timer for 5 minutes",
            "Play some music"
        ]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "device": simulator.device_name}

@app.get("/metrics")
async def metrics():
    return {
        "device": simulator.device_name,
        "timestamp": datetime.now().isoformat(),
        "requests_processed": simulator.requests,
        **simulator.get_metrics()
    }

@app.post("/inference")
async def inference(request: InferenceRequest):
    return await simulator.process_inference(request)

# Test function
async def run_tests():
    """Run automated tests"""
    print("🧪 Running Quick Tests...")
    
    test_queries = [
        "Turn on the living room lights",
        "What's the weather like today?", 
        "Check system status",
        "Hello, how are you?",
        "Analyze sensor data"
    ]
    
    import httpx
    async with httpx.AsyncClient() as client:
        for query in test_queries:
            print(f"\n👤 Query: '{query}'")
            
            try:
                response = await client.post(
                    "http://localhost:8000/inference",
                    json={"query": query},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"🤖 Response: {result['result']}")
                    print(f"⏱️ Time: {result['processing_time_ms']:.1f}ms")
                else:
                    print(f"❌ Error: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Failed: {e}")
            
            await asyncio.sleep(1)

if __name__ == "__main__":
    print("🚀 Starting Quick Raspberry Pi IoT Simulator")
    print("📱 Access at: http://localhost:8000")
    print("🔧 API docs: http://localhost:8000/docs")
    print("📊 Metrics: http://localhost:8000/metrics")
    print("\n💡 Try these commands in another terminal:")
    print("curl http://localhost:8000/")
    print('curl -X POST http://localhost:8000/inference -H "Content-Type: application/json" -d \'{"query": "Turn on the lights"}\'')
    print("\n🛑 Press Ctrl+C to stop")
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
