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
