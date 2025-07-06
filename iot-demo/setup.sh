#!/bin/bash
# Zig AI Platform IoT Demo Setup Script for Raspberry Pi / Linux
# This script sets up the complete IoT demonstration environment

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/setup.log"
SKIP_DOCKER=false
QUICK_SETUP=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    if [ "$VERBOSE" = true ]; then
        echo -e "$1"
    fi
}

# Colored output functions
print_success() { echo -e "${GREEN}$1${NC}"; log "SUCCESS: $1"; }
print_error() { echo -e "${RED}$1${NC}"; log "ERROR: $1"; }
print_warning() { echo -e "${YELLOW}$1${NC}"; log "WARNING: $1"; }
print_info() { echo -e "${CYAN}$1${NC}"; log "INFO: $1"; }
print_header() { 
    echo ""
    echo -e "${PURPLE}$(printf '=%.0s' {1..60})${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}$(printf '=%.0s' {1..60})${NC}"
    echo ""
    log "HEADER: $1"
}

# Help function
show_help() {
    cat << EOF
Zig AI Platform IoT Demo Setup Script

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -s, --skip-docker   Skip Docker installation and setup
    -q, --quick         Quick setup (skip model downloads)
    -v, --verbose       Enable verbose output
    
Examples:
    $0                  # Full setup
    $0 -q               # Quick setup
    $0 -s -v            # Skip Docker with verbose output

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        -q|--quick)
            QUICK_SETUP=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Detect system architecture and OS
detect_system() {
    print_header "Detecting System Information"
    
    ARCH=$(uname -m)
    OS=$(uname -s)
    
    print_info "Architecture: $ARCH"
    print_info "Operating System: $OS"
    
    # Check if running on Raspberry Pi
    if [ -f /proc/device-tree/model ]; then
        PI_MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Unknown Pi")
        print_info "Raspberry Pi Model: $PI_MODEL"
        IS_RASPBERRY_PI=true
    else
        IS_RASPBERRY_PI=false
        print_info "Not running on Raspberry Pi"
    fi
    
    # Check available memory
    TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.1f", $2/1024}')
    print_info "Total Memory: ${TOTAL_MEM}GB"
    
    if (( $(echo "$TOTAL_MEM < 2.0" | bc -l) )); then
        print_warning "Low memory detected. Consider using lighter models."
    fi
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    local missing_deps=()
    
    # Check essential commands
    for cmd in curl wget git python3 pip3; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=($cmd)
        else
            print_success "✓ $cmd is available"
        fi
    done
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "✓ Python $PYTHON_VERSION detected"
        else
            print_warning "⚠ Python 3.8+ recommended (found $PYTHON_VERSION)"
        fi
    fi
    
    # Check Docker if not skipping
    if [ "$SKIP_DOCKER" = false ]; then
        if ! command -v docker &> /dev/null; then
            missing_deps+=(docker)
        else
            if docker ps &> /dev/null; then
                print_success "✓ Docker is running"
            else
                print_warning "⚠ Docker is installed but not running"
            fi
        fi
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df "$SCRIPT_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
    if [ "$AVAILABLE_SPACE" -lt 5 ]; then
        print_warning "⚠ Low disk space: ${AVAILABLE_SPACE}GB available (5GB+ recommended)"
    else
        print_success "✓ Disk space: ${AVAILABLE_SPACE}GB available"
    fi
    
    # Install missing dependencies
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_warning "Missing dependencies: ${missing_deps[*]}"
        install_dependencies "${missing_deps[@]}"
    else
        print_success "✓ All prerequisites met!"
    fi
}

# Install missing dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    local deps=("$@")
    
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        PKG_MANAGER="apt-get"
        UPDATE_CMD="sudo apt-get update"
        INSTALL_CMD="sudo apt-get install -y"
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
        UPDATE_CMD="sudo yum update -y"
        INSTALL_CMD="sudo yum install -y"
    elif command -v pacman &> /dev/null; then
        PKG_MANAGER="pacman"
        UPDATE_CMD="sudo pacman -Sy"
        INSTALL_CMD="sudo pacman -S --noconfirm"
    else
        print_error "❌ No supported package manager found"
        exit 1
    fi
    
    print_info "Using package manager: $PKG_MANAGER"
    
    # Update package lists
    print_info "Updating package lists..."
    $UPDATE_CMD
    
    # Install dependencies
    for dep in "${deps[@]}"; do
        case $dep in
            docker)
                install_docker
                ;;
            pip3)
                $INSTALL_CMD python3-pip
                ;;
            *)
                print_info "Installing $dep..."
                $INSTALL_CMD $dep
                ;;
        esac
    done
}

# Install Docker
install_docker() {
    print_info "Installing Docker..."
    
    if [ "$IS_RASPBERRY_PI" = true ]; then
        # Raspberry Pi specific Docker installation
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        print_warning "⚠ Please log out and back in for Docker group membership to take effect"
    else
        # Standard Docker installation
        case $PKG_MANAGER in
            apt-get)
                sudo apt-get install -y docker.io docker-compose
                sudo systemctl enable docker
                sudo systemctl start docker
                sudo usermod -aG docker $USER
                ;;
            yum)
                sudo yum install -y docker docker-compose
                sudo systemctl enable docker
                sudo systemctl start docker
                sudo usermod -aG docker $USER
                ;;
            *)
                print_error "❌ Docker installation not supported for $PKG_MANAGER"
                exit 1
                ;;
        esac
    fi
    
    print_success "✓ Docker installed successfully"
}

# Install Zig compiler
install_zig() {
    print_header "Installing Zig Compiler"
    
    local zig_version="0.11.0"
    local zig_dir="/opt/zig"
    
    # Determine architecture-specific download
    case $ARCH in
        x86_64)
            zig_arch="x86_64"
            ;;
        aarch64|arm64)
            zig_arch="aarch64"
            ;;
        armv7l)
            zig_arch="armv7a"
            ;;
        *)
            print_error "❌ Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
    
    local zig_url="https://ziglang.org/download/${zig_version}/zig-linux-${zig_arch}-${zig_version}.tar.xz"
    local zig_archive="zig-linux-${zig_arch}-${zig_version}.tar.xz"
    
    print_info "Downloading Zig for $zig_arch..."
    wget -q "$zig_url" -O "$zig_archive"
    
    print_info "Installing Zig to $zig_dir..."
    sudo mkdir -p "$zig_dir"
    sudo tar -xf "$zig_archive" -C "$zig_dir" --strip-components=1
    sudo ln -sf "$zig_dir/zig" /usr/local/bin/zig
    
    # Cleanup
    rm "$zig_archive"
    
    # Verify installation
    if zig version &> /dev/null; then
        print_success "✓ Zig $(zig version) installed successfully"
    else
        print_error "❌ Zig installation failed"
        exit 1
    fi
}

# Create directory structure
create_directories() {
    print_header "Creating Directory Structure"
    
    local directories=(
        "config"
        "src/pi-simulator"
        "src/edge-coordinator"
        "src/iot-client"
        "src/monitoring"
        "docker"
        "models/lightweight-llm-1b"
        "models/industrial-ai-3b"
        "models/retail-ai-2b"
        "scripts"
        "docs"
        "logs"
        "data/sensor-data"
        "data/inference-results"
        "web/dashboard"
    )
    
    for dir in "${directories[@]}"; do
        local full_path="$SCRIPT_DIR/$dir"
        if [ ! -d "$full_path" ]; then
            mkdir -p "$full_path"
            print_success "✓ Created directory: $dir"
        else
            print_info "✓ Directory exists: $dir"
        fi
    done
}

# Create configuration files
create_configuration_files() {
    print_header "Creating Configuration Files"
    
    # Pi devices configuration
    cat > "$SCRIPT_DIR/config/pi-devices.yaml" << 'EOF'
devices:
  - name: "smart-home-pi"
    type: "raspberry-pi-4"
    memory_mb: 4096
    cpu_cores: 4
    architecture: "arm64"
    network:
      type: "wifi"
      bandwidth_mbps: 50
      latency_ms: 20
    location: "living-room"
    use_case: "smart-home"
    
  - name: "industrial-pi"
    type: "raspberry-pi-4"
    memory_mb: 8192
    cpu_cores: 4
    architecture: "arm64"
    network:
      type: "ethernet"
      bandwidth_mbps: 100
      latency_ms: 5
    location: "factory-floor"
    use_case: "industrial-iot"
    
  - name: "retail-pi"
    type: "raspberry-pi-4"
    memory_mb: 4096
    cpu_cores: 4
    architecture: "arm64"
    network:
      type: "4g"
      bandwidth_mbps: 25
      latency_ms: 50
    location: "store-front"
    use_case: "retail-edge"

coordinator:
  port: 8080
  max_devices: 10
  health_check_interval: 30
  load_balancing_strategy: "least_loaded"
EOF
    
    print_success "✓ Created pi-devices.yaml"
    
    # Models configuration
    cat > "$SCRIPT_DIR/config/models.yaml" << 'EOF'
models:
  - name: "lightweight-llm-1b"
    parameters: 1000000000
    quantization: "int8"
    memory_requirement_mb: 1200
    use_cases: ["smart-home", "general-chat"]
    inference_time_ms: 3200
    
  - name: "industrial-ai-3b"
    parameters: 3000000000
    quantization: "int4"
    memory_requirement_mb: 1800
    use_cases: ["industrial-iot", "predictive-maintenance"]
    inference_time_ms: 2800
    
  - name: "retail-ai-2b"
    parameters: 2000000000
    quantization: "int8"
    memory_requirement_mb: 2100
    use_cases: ["retail-edge", "customer-service"]
    inference_time_ms: 4100

optimization:
  enable_model_caching: true
  cache_size_mb: 1024
  enable_dynamic_loading: true
  prefetch_popular_models: true
EOF
    
    print_success "✓ Created models.yaml"
    
    # Python requirements
    cat > "$SCRIPT_DIR/requirements.txt" << 'EOF'
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pyyaml>=6.0.1
psutil>=5.9.0
pydantic>=2.4.0
httpx>=0.25.0
aiofiles>=23.2.1
python-multipart>=0.0.6
EOF
    
    print_success "✓ Created requirements.txt"
}

# Install Python dependencies
install_python_dependencies() {
    print_header "Installing Python Dependencies"
    
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        print_info "Installing Python packages..."
        pip3 install --user -r "$SCRIPT_DIR/requirements.txt"
        print_success "✓ Python dependencies installed"
    else
        print_warning "⚠ requirements.txt not found, skipping Python dependencies"
    fi
}

# Create startup scripts
create_startup_scripts() {
    print_header "Creating Startup Scripts"
    
    # Create start script
    cat > "$SCRIPT_DIR/start-iot-demo.sh" << 'EOF'
#!/bin/bash
# Start IoT Demo Script for Linux/Raspberry Pi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}Starting Zig AI Platform IoT Demo...${NC}"

# Check if Docker is available and running
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        echo -e "${GREEN}✓ Docker is running${NC}"
        
        # Start with Docker Compose
        cd "$SCRIPT_DIR/docker"
        docker-compose up -d
        
        echo ""
        echo -e "${GREEN}🚀 IoT Demo is starting up!${NC}"
        echo -e "${CYAN}📊 Dashboard: http://localhost:8080${NC}"
        echo -e "${CYAN}📈 Monitoring: http://localhost:3000 (admin/admin)${NC}"
        echo ""
        echo -e "${CYAN}Run './run-scenarios.sh' to start test scenarios${NC}"
    else
        echo -e "${RED}❌ Docker is not running. Please start Docker service.${NC}"
        exit 1
    fi
else
    # Start without Docker (direct Python execution)
    echo -e "${CYAN}Starting without Docker...${NC}"
    
    # Start edge coordinator in background
    cd "$SCRIPT_DIR"
    if command -v zig &> /dev/null; then
        echo "Starting edge coordinator..."
        zig run src/edge-coordinator/main.zig &
        COORDINATOR_PID=$!
        echo $COORDINATOR_PID > coordinator.pid
    fi
    
    # Start Pi simulators
    export DEVICE_NAME="smart-home-pi"
    export DEVICE_PORT="8081"
    python3 src/pi-simulator/main.py &
    echo $! > smart-home-pi.pid
    
    export DEVICE_NAME="industrial-pi"
    export DEVICE_PORT="8082"
    python3 src/pi-simulator/main.py &
    echo $! > industrial-pi.pid
    
    export DEVICE_NAME="retail-pi"
    export DEVICE_PORT="8083"
    python3 src/pi-simulator/main.py &
    echo $! > retail-pi.pid
    
    echo -e "${GREEN}✓ Services started${NC}"
    echo -e "${CYAN}📊 Smart Home Pi: http://localhost:8081${NC}"
    echo -e "${CYAN}🏭 Industrial Pi: http://localhost:8082${NC}"
    echo -e "${CYAN}🛒 Retail Pi: http://localhost:8083${NC}"
fi
EOF
    
    chmod +x "$SCRIPT_DIR/start-iot-demo.sh"
    print_success "✓ Created start-iot-demo.sh"
    
    # Create stop script
    cat > "$SCRIPT_DIR/stop-iot-demo.sh" << 'EOF'
#!/bin/bash
# Stop IoT Demo Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Stopping IoT Demo...${NC}"

cd "$SCRIPT_DIR"

# Stop Docker containers if running
if [ -d "docker" ] && command -v docker-compose &> /dev/null; then
    cd docker
    docker-compose down
    cd ..
fi

# Stop direct processes
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

echo -e "${GREEN}✓ IoT Demo stopped${NC}"
EOF
    
    chmod +x "$SCRIPT_DIR/stop-iot-demo.sh"
    print_success "✓ Created stop-iot-demo.sh"
}

# Download or create model placeholders
setup_models() {
    if [ "$QUICK_SETUP" = true ]; then
        print_warning "⚠ Skipping model setup in quick mode"
        return
    fi
    
    print_header "Setting Up AI Models"
    
    local model_dirs=("lightweight-llm-1b" "industrial-ai-3b" "retail-ai-2b")
    
    for model_dir in "${model_dirs[@]}"; do
        local model_path="$SCRIPT_DIR/models/$model_dir"
        
        # Create model metadata
        cat > "$model_path/metadata.json" << EOF
{
    "name": "$model_dir",
    "version": "1.0.0",
    "architecture": "transformer",
    "quantization": "int8",
    "created": "$(date -Iseconds)",
    "size_mb": 1200,
    "description": "Optimized model for IoT edge deployment"
}
EOF
        
        # Create dummy model file
        echo "# Placeholder model file for $model_dir" > "$model_path/model.zig"
        echo "# In production, this would contain the actual model weights" >> "$model_path/model.zig"
        
        print_success "✓ Created placeholder model: $model_dir"
    done
}

# Show completion summary
show_summary() {
    print_header "Setup Complete!"
    
    print_success "🎉 IoT Demo environment has been set up successfully!"
    echo ""
    print_info "Next steps:"
    print_info "1. Start the demo: ./start-iot-demo.sh"
    print_info "2. Run test scenarios: ./run-scenarios.sh"
    print_info "3. Monitor performance: ./monitor-iot.sh"
    echo ""
    print_info "Access points:"
    print_info "• Main Dashboard: http://localhost:8080"
    print_info "• Smart Home Pi: http://localhost:8081"
    print_info "• Industrial Pi: http://localhost:8082"
    print_info "• Retail Pi: http://localhost:8083"
    echo ""
    
    if [ "$IS_RASPBERRY_PI" = true ]; then
        print_info "🍓 Raspberry Pi specific notes:"
        print_info "• Consider enabling GPU memory split for AI workloads"
        print_info "• Monitor temperature during intensive inference"
        print_info "• Use 'vcgencmd measure_temp' to check Pi temperature"
    fi
    
    print_info "For help: ./setup.sh --help"
}

# Main execution
main() {
    print_header "Zig AI Platform IoT Demo Setup"
    
    # Initialize log file
    echo "Setup started at $(date)" > "$LOG_FILE"
    
    detect_system
    check_prerequisites
    install_zig
    create_directories
    create_configuration_files
    install_python_dependencies
    create_startup_scripts
    setup_models
    show_summary
    
    print_success "✅ Setup completed successfully!"
    log "Setup completed successfully"
}

# Run main function
main "$@"
