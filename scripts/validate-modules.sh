#!/usr/bin/env bash
# Validate that all modules can be built
# Reports which modules have missing system dependencies

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Module Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if pkg-config is available
PKG_CONFIG_AVAILABLE=false
if command -v pkg-config &> /dev/null; then
    PKG_CONFIG_AVAILABLE=true
else
    echo -e "${RED}✗ pkg-config not found${NC}"
    echo -e "  Install with: ${YELLOW}sudo apt-get install pkg-config${NC}"
    echo ""
fi

# Use Python to parse the modules index
VALIDATION_RESULT=$(python3 << 'PYTHON_SCRIPT'
import json
import sys
import subprocess

try:
    with open('modules/index.json', 'r') as f:
        data = json.load(f)
except:
    print("ERROR:Could not read modules/index.json")
    sys.exit(1)

modules_with_deps = []
for module in data.get('modules', []):
    deps = module.get('dependencies', {})
    # Handle both dict and list formats
    if isinstance(deps, list):
        continue  # Old format, skip
    system_deps = deps.get('system', [])
    if system_deps and len(system_deps) > 0:
        dep_ids = [dep.get('id', '') for dep in system_deps if 'id' in dep]
        if dep_ids:
            modules_with_deps.append({
                'name': module['name'],
                'deps': dep_ids,
                'install': {dep.get('id'): dep.get('install', {}) for dep in system_deps}
            })

if not modules_with_deps:
    print("NONE")
    sys.exit(0)

# Check each module
available = []
missing = []

for module in modules_with_deps:
    module_ok = True
    missing_deps = []

    for dep in module['deps']:
        # Check if available via pkg-config
        try:
            result = subprocess.run(['pkg-config', '--exists', dep],
                                  capture_output=True, timeout=2)
            if result.returncode != 0:
                module_ok = False
                missing_deps.append(dep)
        except:
            module_ok = False
            missing_deps.append(dep)

    if module_ok:
        available.append(module['name'])
        print(f"OK:{module['name']}")
    else:
        missing.append({
            'name': module['name'],
            'missing': missing_deps,
            'install': module['install']
        })
        print(f"MISSING:{module['name']}:{','.join(missing_deps)}")

# Print summary info
print(f"SUMMARY:{len(available)}:{len(missing)}")
for mod in missing:
    # Detect platform and get install command
    platform = ""
    try:
        if subprocess.run(['command', '-v', 'apt-get'], capture_output=True, shell=True).returncode == 0:
            platform = "apt"
        elif subprocess.run(['command', '-v', 'brew'], capture_output=True, shell=True).returncode == 0:
            platform = "brew"
    except:
        pass

    install_cmds = []
    for dep in mod['missing']:
        if dep in mod['install'] and platform in mod['install'][dep]:
            install_cmds.append(mod['install'][dep][platform])

    if install_cmds:
        print(f"INSTALL:{mod['name']}:{platform}:{' '.join(install_cmds)}")
PYTHON_SCRIPT
)

if echo "$VALIDATION_RESULT" | grep -q "^ERROR:"; then
    echo -e "${RED}Error: $(echo "$VALIDATION_RESULT" | grep "^ERROR:" | cut -d: -f2-)${NC}"
    exit 1
fi

if echo "$VALIDATION_RESULT" | grep -q "^NONE"; then
    echo -e "${GREEN}✓ No modules require system dependencies${NC}"
    exit 0
fi

# Parse results
AVAILABLE_COUNT=$(echo "$VALIDATION_RESULT" | grep "^SUMMARY:" | cut -d: -f2)
MISSING_COUNT=$(echo "$VALIDATION_RESULT" | grep "^SUMMARY:" | cut -d: -f3)
TOTAL=$((AVAILABLE_COUNT + MISSING_COUNT))

echo "Checking system dependencies for modules..."
echo ""

# Print module status
while IFS= read -r line; do
    if [[ $line =~ ^OK: ]]; then
        module=$(echo "$line" | cut -d: -f2)
        echo -e "  ${GREEN}✓${NC} $module"
    elif [[ $line =~ ^MISSING: ]]; then
        module=$(echo "$line" | cut -d: -f2)
        deps=$(echo "$line" | cut -d: -f3)
        echo -e "  ${RED}✗${NC} $module (missing: $deps)"
    fi
done <<< "$VALIDATION_RESULT"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "Total modules with dependencies: ${TOTAL}"
echo -e "${GREEN}Available: ${AVAILABLE_COUNT}${NC}"
echo -e "${RED}Missing dependencies: ${MISSING_COUNT}${NC}"
echo ""

if [ "$MISSING_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}Modules with missing dependencies:${NC}"

    # Get list of missing modules
    MISSING_MODULES=$(echo "$VALIDATION_RESULT" | grep "^MISSING:" | cut -d: -f2 | sort)

    for mod in $MISSING_MODULES; do
        echo -e "  • $mod"

        # Get install command for this module
        INSTALL_LINE=$(echo "$VALIDATION_RESULT" | grep "^INSTALL:$mod:")
        if [ -n "$INSTALL_LINE" ]; then
            PLATFORM=$(echo "$INSTALL_LINE" | cut -d: -f3)
            PACKAGES=$(echo "$INSTALL_LINE" | cut -d: -f4-)

            if [ "$PLATFORM" = "apt" ]; then
                echo -e "    ${BLUE}Install:${NC} sudo apt-get install $PACKAGES"
            elif [ "$PLATFORM" = "brew" ]; then
                echo -e "    ${BLUE}Install:${NC} brew install $PACKAGES"
            fi
        fi
    done
    echo ""
    echo -e "${YELLOW}Note:${NC} Examples requiring these modules will be skipped."
    echo -e "      To build all examples, install the missing dependencies above."
    echo ""
    exit 1
else
    echo -e "${GREEN}✓ All modules have required system dependencies!${NC}"
    echo ""
fi
