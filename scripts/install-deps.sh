#!/usr/bin/env bash
# Install all missing module dependencies
# Must be run with sudo on systems that require it

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Installing Module Dependencies${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running as root or with sudo cached
if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
    echo -e "${RED}Error: This script requires sudo access${NC}"
    echo -e "Please run with: ${YELLOW}sudo make install-deps${NC}"
    echo ""
    exit 1
fi

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MGR="apt"
elif command -v brew &> /dev/null; then
    PKG_MGR="brew"
elif command -v pkg &> /dev/null; then
    PKG_MGR="pkg"
elif command -v dnf &> /dev/null; then
    PKG_MGR="dnf"
elif command -v pacman &> /dev/null; then
    PKG_MGR="pacman"
else
    echo -e "${RED}Error: No supported package manager found${NC}"
    exit 1
fi

echo -e "Detected package manager: ${GREEN}$PKG_MGR${NC}"
echo ""

# Parse modules and collect packages to install
PACKAGES_TO_INSTALL=$(python3 << 'PYTHON_SCRIPT'
import json
import sys
import os

try:
    with open('modules/index.json', 'r') as f:
        data = json.load(f)
except:
    print("ERROR:Could not read modules/index.json", file=sys.stderr)
    sys.exit(1)

pkg_mgr = sys.argv[1] if len(sys.argv) > 1 else "apt"
packages = set()

# Method 1: Check system dependencies in modules/index.json (old format)
for module in data.get('modules', []):
    deps = module.get('dependencies', {})
    if isinstance(deps, list):
        continue

    system_deps = deps.get('system', [])
    for dep in system_deps:
        if 'install' in dep and pkg_mgr in dep['install']:
            pkg_name = dep['install'][pkg_mgr]
            packages.add(pkg_name)

# Method 2: Check individual module.json files for install field (new format)
for module in data.get('modules', []):
    module_name = module['name']
    module_path = f'modules/{module_name}/module.json'

    if not os.path.exists(module_path):
        continue

    try:
        with open(module_path) as f:
            mod_manifest = json.load(f)
    except:
        continue

    # Get install information
    install_info = mod_manifest.get('install', {})

    # Check for platform-specific install info
    if 'linux' in install_info and pkg_mgr == 'apt':
        if 'apt' in install_info['linux']:
            packages.add(install_info['linux']['apt'])
    elif 'macos' in install_info and pkg_mgr == 'brew':
        if 'brew' in install_info['macos']:
            packages.add(install_info['macos']['brew'])

# Print unique packages
for pkg in sorted(packages):
    print(pkg)
PYTHON_SCRIPT
)

if [ -z "$PACKAGES_TO_INSTALL" ]; then
    echo -e "${GREEN}✓ No system dependencies required${NC}"
    exit 0
fi

# Show what will be installed
echo -e "Packages to install:"
echo "$PACKAGES_TO_INSTALL" | while read -r pkg; do
    echo -e "  • $pkg"
done
echo ""

# Install based on package manager
case "$PKG_MGR" in
    apt)
        echo -e "${BLUE}Updating package lists...${NC}"
        sudo apt-get update -qq

        echo -e "${BLUE}Installing packages...${NC}"
        DEBIAN_FRONTEND=noninteractive sudo apt-get install -y $PACKAGES_TO_INSTALL
        ;;

    brew)
        echo -e "${BLUE}Installing packages...${NC}"
        for pkg in $PACKAGES_TO_INSTALL; do
            if brew list "$pkg" &>/dev/null; then
                echo "  ✓ $pkg already installed"
            else
                brew install "$pkg"
            fi
        done
        ;;

    pkg)
        echo -e "${BLUE}Installing packages...${NC}"
        sudo pkg install -y $PACKAGES_TO_INSTALL
        ;;

    dnf)
        echo -e "${BLUE}Installing packages...${NC}"
        sudo dnf install -y $PACKAGES_TO_INSTALL
        ;;

    pacman)
        echo -e "${BLUE}Installing packages...${NC}"
        sudo pacman -S --noconfirm $PACKAGES_TO_INSTALL
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Dependencies Installed Successfully${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Now you can build everything:"
echo -e "  ${YELLOW}make examples${NC}"
echo ""
