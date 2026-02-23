#!/bin/bash
# TWAE_MMD VS Code Docker Build Script
# Builds the Docker image with VS Code Server and all dependencies

set -e  # Exit on any error

echo "🐳 Building TWAE_MMD VS Code Docker Environment..."
echo "=================================================="

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

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}❌ Dockerfile not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Build Information:${NC}"
echo "   • Image Name: twae_vscode:latest"
echo "   • Base Image: tensorflow/tensorflow:2.13.0-gpu"
echo "   • VS Code Server: Latest"
echo "   • Python Version: 3.11"
echo "   • GPU Support: NVIDIA RTX 2060"
echo ""

# Clean up any existing containers/images if requested
read -p "🧹 Clean up existing images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🧹 Cleaning up existing containers and images...${NC}"
    docker stop twae_mmd_vscode 2>/dev/null || true
    docker rm twae_mmd_vscode 2>/dev/null || true
    docker rmi twae_vscode:latest 2>/dev/null || true
    docker system prune -f
fi

# Build the Docker image
echo -e "${BLUE}🔨 Building Docker image...${NC}"
echo "This may take 15-20 minutes for the first build..."
echo ""

start_time=$(date +%s)

if docker build -t twae_vscode:latest .; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    echo ""
    echo -e "${GREEN}✅ Build completed successfully!${NC}"
    echo -e "${GREEN}⏱️  Build time: ${minutes}m ${seconds}s${NC}"
    echo ""
    echo -e "${BLUE}📊 Image Information:${NC}"
    docker images twae_vscode:latest
    echo ""
    echo -e "${GREEN}🚀 Ready to start! Run: ./scripts/run.sh${NC}"
else
    echo -e "${RED}❌ Build failed! Check the error messages above.${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📝 Next Steps:${NC}"
echo "   1. Run: ./scripts/run.sh"
echo "   2. Access VS Code at: http://localhost:8080"
echo "   3. Password: twae_mmd_research"
echo "   4. Start developing your TWAE_MMD research!"
echo ""

