#!/bin/bash
# TWAE_MMD VS Code Docker Shell Access Script
# Provides shell access to the running VS Code container

set -e  # Exit on any error

echo "🐚 Accessing TWAE_MMD VS Code Container Shell..."
echo "=============================================="

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

# Check if container is running
if ! docker ps -q -f name=twae_mmd_vscode | grep -q .; then
    echo -e "${RED}❌ Container 'twae_mmd_vscode' is not running.${NC}"
    echo -e "${YELLOW}💡 Please run: ./scripts/run.sh first${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Container Information:${NC}"
docker ps -f name=twae_mmd_vscode --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

echo -e "${GREEN}🔗 Connecting to container shell...${NC}"
echo -e "${BLUE}💡 You are now inside the VS Code container as user 'vscode'${NC}"
echo -e "${BLUE}💡 Working directory: /workspace/TWAE_AMP_Generation${NC}"
echo -e "${BLUE}💡 Type 'exit' to return to host${NC}"
echo ""

# Access container shell as vscode user
docker exec -it twae_mmd_vscode bash

echo ""
echo -e "${GREEN}✅ Shell session ended.${NC}"

