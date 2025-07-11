#!/bin/bash
# Setup Real LLM for IoT Demo
# Quick setup script to get TinyLLM working with the IoT demo

set -e

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we're in the right directory
if [ ! -f "$SCRIPT_DIR/README.md" ]; then
    print_error "Please run this script from the iot-demo directory"
    exit 1
fi

print_header "Real LLM IoT Demo Setup"

# Step 1: Install Python dependencies
print_info "📦 Installing Python dependencies..."
pip3 install --user torch transformers tokenizers accelerate sentencepiece protobuf

# Step 2: Download a lightweight model
print_info "📥 Downloading TinyLLM model..."
chmod +x download-tinyllm.sh
./download-tinyllm.sh qwen1.5-0.5b

# Step 3: Test the LLM
print_info "🧪 Testing LLM inference..."
python3 src/llm/tinyllm_inference.py

# Step 4: Create startup script for real LLM
print_info "📝 Creating startup script..."

cat > start-real-llm-demo.sh << 'EOF'
#!/bin/bash
# Start Real LLM IoT Demo

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting Real LLM IoT Demo...${NC}"

# Start devices with real LLM
echo "Starting Smart Home Pi with real LLM..."
USE_REAL_LLM=true MODEL_NAME=qwen1.5-0.5b DEVICE_NAME=smart-home-pi DEVICE_PORT=8081 python3 src/pi-simulator/real_llm_pi.py &
PID1=$!

echo "Starting Industrial Pi with real LLM..."
USE_REAL_LLM=true MODEL_NAME=qwen1.5-0.5b DEVICE_NAME=industrial-pi DEVICE_PORT=8082 python3 src/pi-simulator/real_llm_pi.py &
PID2=$!

echo "Starting Retail Pi with real LLM..."
USE_REAL_LLM=true MODEL_NAME=qwen1.5-0.5b DEVICE_NAME=retail-pi DEVICE_PORT=8083 python3 src/pi-simulator/real_llm_pi.py &
PID3=$!

# Save PIDs
echo $PID1 > smart-home-llm.pid
echo $PID2 > industrial-llm.pid
echo $PID3 > retail-llm.pid

echo ""
echo -e "${GREEN}✅ Real LLM devices started!${NC}"
echo ""
echo -e "${CYAN}🤖 Real AI Inference Available At:${NC}"
echo -e "${CYAN}   🏠 Smart Home Pi: http://localhost:8081${NC}"
echo -e "${CYAN}   🏭 Industrial Pi: http://localhost:8082${NC}"
echo -e "${CYAN}   🛒 Retail Pi: http://localhost:8083${NC}"
echo ""
echo -e "${CYAN}💡 Note: First inference may take longer as models load${NC}"
echo -e "${CYAN}🛑 Stop: ./stop-real-llm-demo.sh${NC}"

# Wait for services to start
sleep 5

# Test one device
echo ""
echo -e "${CYAN}🧪 Testing real LLM inference...${NC}"
curl -X POST http://localhost:8081/api/inference \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, this is a test of real AI on Raspberry Pi!"}' \
  2>/dev/null | python3 -m json.tool || echo "Service starting up..."
EOF

chmod +x start-real-llm-demo.sh

# Step 5: Create stop script
cat > stop-real-llm-demo.sh << 'EOF'
#!/bin/bash
# Stop Real LLM IoT Demo

echo "🛑 Stopping Real LLM IoT Demo..."

# Kill processes
for pidfile in *-llm.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "Stopped process $pid"
        fi
        rm "$pidfile"
    fi
done

echo "✅ Real LLM demo stopped"
EOF

chmod +x stop-real-llm-demo.sh

# Step 6: Create test script
cat > test-real-llm.py << 'EOF'
#!/usr/bin/env python3
"""
Test Real LLM IoT Demo
"""

import asyncio
import httpx
import json
import time

async def test_real_llm():
    """Test real LLM inference on all devices"""
    
    devices = [
        {"name": "Smart Home Pi", "port": 8081, "icon": "🏠"},
        {"name": "Industrial Pi", "port": 8082, "icon": "🏭"},
        {"name": "Retail Pi", "port": 8083, "icon": "🛒"}
    ]
    
    test_queries = [
        "Turn on the living room lights",
        "What is the current temperature?",
        "Check system status",
        "Hello, how are you today?"
    ]
    
    print("🧪 Testing Real LLM IoT Demo")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for device in devices:
            print(f"\n{device['icon']} Testing {device['name']} (Port {device['port']})")
            print("-" * 40)
            
            # Check if device is ready
            try:
                health = await client.get(f"http://localhost:{device['port']}/health")
                if health.status_code == 200:
                    print("✅ Device online")
                else:
                    print("❌ Device not responding")
                    continue
            except:
                print("❌ Device not reachable")
                continue
            
            # Test inference
            for query in test_queries[:2]:  # Test 2 queries per device
                print(f"\n👤 Query: '{query}'")
                
                try:
                    start_time = time.time()
                    response = await client.post(
                        f"http://localhost:{device['port']}/api/inference",
                        json={"query": query}
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"🤖 Response: {result['result']}")
                        print(f"⏱️ Time: {result['processing_time_ms']:.1f}ms")
                        print(f"🧠 Model: {result['model_used']}")
                        print(f"🔧 Type: {result['inference_type']}")
                        
                        if result['inference_type'] == 'real_llm':
                            print("✅ Real LLM inference working!")
                        else:
                            print("⚠️ Using fallback (LLM not loaded)")
                    else:
                        print(f"❌ Error: {response.status_code}")
                        
                except Exception as e:
                    print(f"❌ Failed: {e}")
                
                await asyncio.sleep(2)  # Rate limiting
    
    print("\n🎉 Real LLM test completed!")
    print("\n💡 Tips:")
    print("   • First inference may be slower (model loading)")
    print("   • Check device dashboards at http://localhost:808X")
    print("   • Monitor resource usage during inference")

if __name__ == "__main__":
    asyncio.run(test_real_llm())
EOF

chmod +x test-real-llm.py

print_header "Setup Complete!"

print_success "🎉 Real LLM IoT Demo is ready!"
echo ""
print_info "Quick start:"
print_info "1. ./start-real-llm-demo.sh    # Start with real AI"
print_info "2. python3 test-real-llm.py    # Test real LLM inference"
print_info "3. Open http://localhost:8081   # Smart Home dashboard"
print_info "4. ./stop-real-llm-demo.sh     # Stop demo"
echo ""
print_info "🤖 Features:"
print_info "   • Real TinyLLM inference (Qwen 1.5 0.5B)"
print_info "   • ~1GB RAM usage per device"
print_info "   • 1-5 second response times"
print_info "   • Interactive web dashboards"
print_info "   • Fallback to simulation if model fails"
echo ""
print_warning "⚠️ Note: First inference takes longer (model loading)"
print_info "💡 For faster startup, models are loaded asynchronously"
