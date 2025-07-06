# Online Raspberry Pi Simulator Guide

## 🌐 Run the Zig AI Platform IoT Demo in Online Environments

This guide shows you how to run our IoT demo in various online platforms without needing physical Raspberry Pi hardware.

## 🚀 Quick Start Options

### Option 1: GitHub Codespaces (Recommended)

1. **Fork the Repository**
   - Go to: https://github.com/anachary/zig-ai-platform
   - Click "Fork" to create your copy

2. **Open in Codespaces**
   - Click the green "Code" button
   - Select "Codespaces" tab
   - Click "Create codespace on main"

3. **Run the Demo**
   ```bash
   cd iot-demo
   chmod +x online-simulator-setup.sh
   ./online-simulator-setup.sh
   ./start_online_demo.sh
   ```

4. **Access the Demo**
   - Smart Home Pi: `http://localhost:8081`
   - Industrial Pi: `http://localhost:8082`
   - Retail Pi: `http://localhost:8083`

### Option 2: Replit

1. **Create New Repl**
   - Go to: https://replit.com
   - Click "Create Repl"
   - Choose "Python" template

2. **Upload Files**
   - Upload the `iot-demo` folder
   - Or clone: `git clone https://github.com/anachary/zig-ai-platform.git`

3. **Run Setup**
   ```bash
   cd iot-demo
   bash online-simulator-setup.sh
   bash start_online_demo.sh
   ```

### Option 3: GitPod

1. **Open in GitPod**
   - Go to: `https://gitpod.io/#https://github.com/anachary/zig-ai-platform`

2. **Run Demo**
   ```bash
   cd iot-demo
   ./online-simulator-setup.sh
   ./start_online_demo.sh
   ```

## 🧪 Testing the Demo

### Automated Tests
```bash
# Run comprehensive test suite
python3 test_online_demo.py
```

### Manual Testing

#### 1. Health Checks
```bash
curl http://localhost:8081/health
curl http://localhost:8082/health  
curl http://localhost:8083/health
```

#### 2. Device Status
```bash
curl http://localhost:8081/api/status
```

#### 3. AI Inference
```bash
curl -X POST http://localhost:8081/api/inference \
  -H "Content-Type: application/json" \
  -d '{"query": "Turn on the living room lights"}'
```

### Interactive Web Interface

Open any device URL in your browser:
- `http://localhost:8081` - Smart Home Pi
- `http://localhost:8082` - Industrial Pi  
- `http://localhost:8083` - Retail Pi

Each provides:
- 📊 Real-time status dashboard
- 🤖 Interactive AI chat interface
- 📈 Performance metrics
- 🔧 API documentation at `/docs`

## 🎯 Demo Scenarios

### Smart Home Assistant
```bash
curl -X POST http://localhost:8081/api/inference \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather like today?"}'
```

**Expected Response:**
```json
{
  "result": "🌤️ Current weather: 72°F, partly cloudy. Perfect day for IoT demos!",
  "processing_time_ms": 1847.3,
  "device_name": "smart-home-pi",
  "model_used": "edge-optimized-llm-online"
}
```

### Industrial Monitoring
```bash
curl -X POST http://localhost:8082/api/inference \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze vibration data: 12.5Hz, 0.8mm amplitude"}'
```

### Retail Customer Service
```bash
curl -X POST http://localhost:8083/api/inference \
  -H "Content-Type: application/json" \
  -d '{"query": "Where can I find organic vegetables?"}'
```

## 📊 Performance Monitoring

### Real-time Metrics
```bash
# Get device metrics
curl http://localhost:8081/api/metrics

# Monitor all devices
for port in 8081 8082 8083; do
  echo "=== Device on port $port ==="
  curl -s http://localhost:$port/api/metrics | jq '.device_name, .cpu_usage_percent, .memory_usage_percent'
done
```

### System Resources
The simulator shows real metrics from the online environment:
- **CPU Usage**: Actual CPU utilization
- **Memory Usage**: Real memory consumption  
- **Disk Usage**: Actual disk space
- **Temperature**: Simulated (45-60°C range)

## 🔧 Customization

### Environment Variables
```bash
# Customize device configuration
export DEVICE_NAME="my-custom-pi"
export DEVICE_PORT="9000"
python3 src/pi-simulator/simple_pi.py
```

### Custom Responses
Edit `src/pi-simulator/simple_pi.py` to add custom AI responses:

```python
def generate_response(self, query: str) -> str:
    if "custom_keyword" in query.lower():
        return "Your custom response here!"
    # ... existing logic
```

## 🌟 Features Demonstrated

### ✅ Edge AI Capabilities
- Lightweight LLM inference simulation
- Context-aware responses
- Resource monitoring
- Performance optimization

### ✅ IoT Integration
- Multiple device coordination
- RESTful API interfaces
- Health monitoring
- Fault tolerance

### ✅ Real-world Scenarios
- Smart home automation
- Industrial sensor analysis
- Retail customer service
- Load balancing

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port
lsof -i :8081

# Kill process
kill -9 <PID>

# Or use different ports
DEVICE_PORT=9081 python3 src/pi-simulator/simple_pi.py
```

### Dependencies Missing
```bash
# Reinstall dependencies
pip3 install --user fastapi uvicorn pyyaml psutil pydantic httpx

# Or use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Slow Response Times
- **Codespaces**: Normal, shared resources
- **Replit**: May have rate limiting
- **Local**: Should be fast

### Memory Issues
```bash
# Check available memory
free -h

# Monitor process memory
ps aux | grep python3
```

## 📱 Mobile Access

If your online environment supports it:
1. Get the public URL (Codespaces/Replit provides this)
2. Access from mobile browser
3. Test voice commands (browser dependent)

## 🔄 Continuous Development

### Auto-restart on Changes
```bash
# Install watchdog
pip3 install watchdog

# Auto-restart server
watchmedo auto-restart --patterns="*.py" --recursive python3 src/pi-simulator/simple_pi.py
```

### Live Debugging
```bash
# Enable debug mode
export DEBUG=true
python3 src/pi-simulator/simple_pi.py
```

## 🎓 Educational Use

This online simulator is perfect for:
- **Learning IoT concepts** without hardware
- **Prototyping AI applications** quickly  
- **Demonstrating edge computing** principles
- **Teaching distributed systems** concepts

## 🚀 Next Steps

1. **Experiment** with different queries
2. **Modify** the AI responses
3. **Add** new device types
4. **Scale** to more devices
5. **Deploy** to real Raspberry Pi hardware

## 📞 Support

- **Issues**: Open GitHub issue
- **Questions**: Check documentation
- **Contributions**: Submit pull request

---

**Happy IoT Development! 🎉**
