#!/bin/bash
# Simplified IoT Demo Setup for Online Simulators
# Compatible with GitHub Codespaces, Replit, and other cloud environments

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_success() { echo -e "${GREEN}$1${NC}"; }
print_error() { echo -e "${RED}$1${NC}"; }
print_warning() { echo -e "${YELLOW}$1${NC}"; }
print_info() { echo -e "${CYAN}$1${NC}"; }

print_header() {
    echo ""
    echo -e "${CYAN}$(printf '=%.0s' {1..50})${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' {1..50})${NC}"
    echo ""
}

# Detect environment
detect_environment() {
    print_header "Detecting Environment"
    
    if [ -n "$CODESPACES" ]; then
        ENV_TYPE="GitHub Codespaces"
        print_info "Running in: $ENV_TYPE"
    elif [ -n "$REPL_ID" ]; then
        ENV_TYPE="Replit"
        print_info "Running in: $ENV_TYPE"
    elif [ -f /.dockerenv ]; then
        ENV_TYPE="Docker Container"
        print_info "Running in: $ENV_TYPE"
    else
        ENV_TYPE="Generic Linux"
        print_info "Running in: $ENV_TYPE"
    fi
    
    # System info
    print_info "Architecture: $(uname -m)"
    print_info "OS: $(uname -s)"
    print_info "Memory: $(free -h | awk 'NR==2{print $2}')"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    # Update package lists
    if command -v apt-get &> /dev/null; then
        print_info "Updating package lists..."
        sudo apt-get update -qq
        
        print_info "Installing Python dependencies..."
        sudo apt-get install -y python3 python3-pip python3-venv curl wget
    fi
    
    # Install Python packages
    print_info "Installing Python packages..."
    pip3 install --user fastapi uvicorn pyyaml psutil pydantic httpx
    
    print_success "✓ Dependencies installed"
}

# Create simplified Pi simulator
create_pi_simulator() {
    print_header "Creating Raspberry Pi Simulator"
    
    mkdir -p src/pi-simulator
    
    cat > src/pi-simulator/simple_pi.py << 'EOF'
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
EOF
    
    chmod +x src/pi-simulator/simple_pi.py
    print_success "✓ Created simplified Pi simulator"
}

# Create test script
create_test_script() {
    print_header "Creating Test Script"
    
    cat > test_online_demo.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for online IoT demo
"""

import asyncio
import httpx
import json
import time
from datetime import datetime

async def test_device(port: int, device_name: str):
    """Test a single device"""
    base_url = f"http://localhost:{port}"
    
    print(f"\n🧪 Testing {device_name} on port {port}")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        try:
            # Health check
            response = await client.get(f"{base_url}/health", timeout=5.0)
            if response.status_code == 200:
                print(f"✅ Health check: OK")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return
            
            # Get status
            response = await client.get(f"{base_url}/api/status", timeout=5.0)
            if response.status_code == 200:
                status = response.json()
                print(f"📊 Status: {status['status']}")
                print(f"⏰ Uptime: {status['uptime_seconds']:.1f}s")
                print(f"💾 Memory: {status['memory_usage_percent']:.1f}%")
                print(f"⚡ CPU: {status['cpu_usage_percent']:.1f}%")
            
            # Test inference
            test_queries = [
                "Turn on the lights",
                "What's the temperature?",
                "Check system status",
                "Hello from online simulator!"
            ]
            
            print(f"\n🤖 Testing AI Inference:")
            for query in test_queries:
                print(f"\n👤 Query: '{query}'")
                
                start_time = time.time()
                response = await client.post(
                    f"{base_url}/api/inference",
                    json={"query": query, "device_id": device_name},
                    timeout=15.0
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"🤖 Response: {result['result']}")
                    print(f"⏱️ Time: {result['processing_time_ms']:.1f}ms")
                    print(f"🧠 Model: {result['model_used']}")
                else:
                    print(f"❌ Inference failed: {response.status_code}")
                
                await asyncio.sleep(1)  # Rate limiting
                
        except Exception as e:
            print(f"❌ Error testing {device_name}: {e}")

async def main():
    """Main test function"""
    print("🚀 Starting Online IoT Demo Test")
    print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test devices
    devices = [
        (8081, "Smart Home Pi"),
        (8082, "Industrial Pi"),
        (8083, "Retail Pi")
    ]
    
    for port, name in devices:
        await test_device(port, name)
    
    print("\n🎉 Test completed!")
    print("\n💡 Next steps:")
    print("   • Open http://localhost:8081 in your browser")
    print("   • Try the API docs at http://localhost:8081/docs")
    print("   • Send custom queries to test AI responses")

if __name__ == "__main__":
    asyncio.run(main())
EOF
    
    chmod +x test_online_demo.py
    print_success "✓ Created test script"
}

# Create startup script
create_startup_script() {
    print_header "Creating Startup Script"
    
    cat > start_online_demo.sh << 'EOF'
#!/bin/bash
# Start Online IoT Demo

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Online IoT Demo...${NC}"

# Start multiple Pi simulators
echo "Starting Smart Home Pi on port 8081..."
DEVICE_NAME="smart-home-pi" DEVICE_PORT=8081 python3 src/pi-simulator/simple_pi.py &
PID1=$!

echo "Starting Industrial Pi on port 8082..."
DEVICE_NAME="industrial-pi" DEVICE_PORT=8082 python3 src/pi-simulator/simple_pi.py &
PID2=$!

echo "Starting Retail Pi on port 8083..."
DEVICE_NAME="retail-pi" DEVICE_PORT=8083 python3 src/pi-simulator/simple_pi.py &
PID3=$!

# Save PIDs for cleanup
echo $PID1 > smart-home-pi.pid
echo $PID2 > industrial-pi.pid
echo $PID3 > retail-pi.pid

echo ""
echo -e "${GREEN}✅ All devices started!${NC}"
echo ""
echo -e "${CYAN}📱 Access Points:${NC}"
echo -e "${CYAN}   🏠 Smart Home Pi: http://localhost:8081${NC}"
echo -e "${CYAN}   🏭 Industrial Pi: http://localhost:8082${NC}"
echo -e "${CYAN}   🛒 Retail Pi: http://localhost:8083${NC}"
echo ""
echo -e "${CYAN}🧪 Run tests: python3 test_online_demo.py${NC}"
echo -e "${CYAN}🛑 Stop demo: ./stop_online_demo.sh${NC}"

# Wait a moment for services to start
sleep 3

# Run initial test
echo ""
echo -e "${CYAN}Running initial connectivity test...${NC}"
python3 test_online_demo.py
EOF
    
    chmod +x start_online_demo.sh
    
    # Create stop script
    cat > stop_online_demo.sh << 'EOF'
#!/bin/bash
# Stop Online IoT Demo

echo "🛑 Stopping Online IoT Demo..."

# Kill processes
for pidfile in *.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "Stopped process $pid"
        fi
        rm "$pidfile"
    fi
done

echo "✅ Demo stopped"
EOF
    
    chmod +x stop_online_demo.sh
    print_success "✓ Created startup scripts"
}

# Main execution
main() {
    print_header "Online IoT Demo Setup"
    
    detect_environment
    install_dependencies
    create_pi_simulator
    create_test_script
    create_startup_script
    
    print_header "Setup Complete!"
    print_success "🎉 Online IoT Demo ready!"
    echo ""
    print_info "Quick start:"
    print_info "1. ./start_online_demo.sh    # Start all simulators"
    print_info "2. Open http://localhost:8081 # Access Smart Home Pi"
    print_info "3. python3 test_online_demo.py # Run comprehensive tests"
    print_info "4. ./stop_online_demo.sh     # Stop all simulators"
    echo ""
    print_info "🌐 This demo works in:"
    print_info "   • GitHub Codespaces"
    print_info "   • Replit"
    print_info "   • Any Linux environment"
    print_info "   • Local development"
}

main "$@"

