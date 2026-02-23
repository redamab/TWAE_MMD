#!/bin/bash
# TWAE_MMD VS Code Docker Run Script
# Starts the VS Code development environment

set -e  # Exit on any error

echo "🚀 Starting TWAE_MMD VS Code Development Environment..."
echo "====================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

# Check if image exists
if ! docker image inspect twae_vscode:latest > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker image 'twae_vscode:latest' not found.${NC}"
    echo -e "${YELLOW}💡 Please run: ./scripts/build.sh first${NC}"
    exit 1
fi

# Stop existing container if running
if docker ps -q -f name=twae_mmd_vscode | grep -q .; then
    echo -e "${YELLOW}⏹️  Stopping existing container...${NC}"
    docker stop twae_mmd_vscode
fi

# Remove existing container if exists
if docker ps -aq -f name=twae_mmd_vscode | grep -q .; then
    echo -e "${YELLOW}🗑️  Removing existing container...${NC}"
    docker rm twae_mmd_vscode
fi

echo -e "${BLUE}📋 Container Configuration:${NC}"
echo "   • Container Name: twae_mmd_vscode"
echo "   • VS Code Server: http://localhost:8080"
echo "   • TensorBoard: http://localhost:6006"
echo "   • Jupyter: http://localhost:8888"
echo "   • Password: twae_mmd_research"
echo "   • GPU Support: Enabled (RTX 2060)"
echo ""

# Start the container
echo -e "${BLUE}🐳 Starting VS Code container...${NC}"

docker run -d \
    --name twae_mmd_vscode \
    --hostname twae-vscode-container \
    --gpus all \
    -p 8080:8080 \
    -p 6006:6006 \
    -p 8888:8888 \
    -v "$(pwd):/workspace/TWAE_AMP_Generation" \
    -v twae_vscode_data:/home/vscode/.local \
    -v twae_data:/workspace/data \
    -v twae_models:/workspace/models \
    -v twae_results:/workspace/results \
    -v twae_logs:/workspace/logs \
    -e PYTHONPATH=/workspace/TWAE_AMP_Generation \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e PASSWORD=twae_mmd_research \
    --restart unless-stopped \
    twae_vscode:latest

# Wait for container to start
echo -e "${YELLOW}⏳ Waiting for VS Code Server to start...${NC}"
sleep 10

# Check if container is running
if docker ps -q -f name=twae_mmd_vscode | grep -q .; then
    echo ""
    echo -e "${GREEN}✅ VS Code environment started successfully!${NC}"
    echo ""
    echo -e "${BLUE}🔗 Access Information:${NC}"
    echo "   • VS Code Server: http://localhost:8080"
    echo "   • Password: twae_mmd_research"
    echo "   • TensorBoard: http://localhost:6006"
    echo "   • Jupyter: http://localhost:8888"
    echo ""
    echo -e "${BLUE}💻 Development Features:${NC}"
    echo "   • Full VS Code IDE with extensions"
    echo "   • Python IntelliSense and debugging"
    echo "   • Jupyter notebook support"
    echo "   • Git integration"
    echo "   • TensorFlow 2.13.0 with GPU support"
    echo "   • All TWAE_MMD dependencies installed"
    echo ""
    echo -e "${GREEN}🎉 Ready for TWAE_MMD research development!${NC}"
    echo ""
    echo -e "${BLUE}📝 Useful Commands:${NC}"
    echo "   • View logs: docker logs twae_mmd_vscode"
    echo "   • Access shell: ./scripts/shell.sh"
    echo "   • Stop container: ./scripts/stop.sh"
    echo ""
    
    # Show container status
    echo -e "${BLUE}📊 Container Status:${NC}"
    docker ps -f name=twae_mmd_vscode --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
else
    echo -e "${RED}❌ Failed to start container. Checking logs...${NC}"
    docker logs twae_mmd_vscode
    exit 1
fi

