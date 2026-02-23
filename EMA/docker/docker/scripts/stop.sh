#!/bin/bash
# TWAE_MMD VS Code Docker Stop Script
# Gracefully stops the VS Code development environment

set -e  # Exit on any error

echo "⏹️  Stopping TWAE_MMD VS Code Development Environment..."
echo "===================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running.${NC}"
    exit 1
fi

# Check if container exists and is running
if docker ps -q -f name=twae_mmd_vscode | grep -q .; then
    echo -e "${YELLOW}⏹️  Stopping VS Code container...${NC}"
    
    # Graceful stop with timeout
    if docker stop twae_mmd_vscode --time 30; then
        echo -e "${GREEN}✅ Container stopped successfully.${NC}"
    else
        echo -e "${RED}❌ Failed to stop container gracefully. Force stopping...${NC}"
        docker kill twae_mmd_vscode
    fi
    
elif docker ps -aq -f name=twae_mmd_vscode | grep -q .; then
    echo -e "${YELLOW}ℹ️  Container exists but is not running.${NC}"
else
    echo -e "${YELLOW}ℹ️  No container named 'twae_mmd_vscode' found.${NC}"
    exit 0
fi

# Option to remove container
echo ""
read -p "🗑️  Remove container completely? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🗑️  Removing container...${NC}"
    docker rm twae_mmd_vscode
    echo -e "${GREEN}✅ Container removed.${NC}"
    
    # Option to remove volumes
    echo ""
    read -p "🗑️  Remove data volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🗑️  Removing volumes...${NC}"
        docker volume rm twae_vscode_data twae_data twae_models twae_results twae_logs 2>/dev/null || true
        echo -e "${GREEN}✅ Volumes removed.${NC}"
    fi
fi

echo ""
echo -e "${BLUE}📊 Current Docker Status:${NC}"
docker ps -a -f name=twae_mmd_vscode --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers found."

echo ""
echo -e "${GREEN}✅ Stop operation completed.${NC}"
echo ""
echo -e "${BLUE}📝 To restart:${NC}"
echo "   • Run: ./scripts/run.sh"
echo ""

