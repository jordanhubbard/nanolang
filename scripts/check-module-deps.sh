#!/usr/bin/env bash
# Module Dependency Checker for Nanolang
# Detects OS, checks module dependencies, and optionally installs them

# Note: Don't use 'set -e' because we want to continue checking all modules
# even if some have missing dependencies

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Mode: check (default) or install (interactive)
MODE="${1:-check}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODULES_DIR="$PROJECT_ROOT/modules"

# Track statistics
TOTAL_MODULES=0
MODULES_WITH_DEPS=0
MISSING_DEPS=0
AVAILABLE_DEPS=0

# Detect OS and package manager
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
        PKG_CHECK_CMD="brew list"
        PKG_INSTALL_CMD="brew install"
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        case "$ID" in
            ubuntu|debian|mint|pop)
                OS="debian"
                PKG_MANAGER="apt"
                PKG_CHECK_CMD="dpkg -l"
                PKG_INSTALL_CMD="sudo apt-get install -y"
                ;;
            fedora|rhel|centos|rocky|almalinux)
                OS="fedora"
                PKG_MANAGER="dnf"
                PKG_CHECK_CMD="rpm -q"
                PKG_INSTALL_CMD="sudo dnf install -y"
                ;;
            arch|manjaro)
                OS="arch"
                PKG_MANAGER="pacman"
                PKG_CHECK_CMD="pacman -Q"
                PKG_INSTALL_CMD="sudo pacman -S --noconfirm"
                ;;
            *)
                OS="unknown"
                PKG_MANAGER="unknown"
                ;;
        esac
    else
        OS="unknown"
        PKG_MANAGER="unknown"
    fi
}

# Map pkg-config names to OS-specific package names
# Usage: map_package "sdl2" -> "sdl2" (homebrew) or "libsdl2-dev" (debian)
map_package() {
    local pkg_config_name="$1"
    local pkg_name=""
    
    case "$OS" in
        macos)
            case "$pkg_config_name" in
                sdl2) pkg_name="sdl2" ;;
                SDL2_ttf) pkg_name="sdl2_ttf" ;;
                SDL2_mixer) pkg_name="sdl2_mixer" ;;
                SDL2_image) pkg_name="sdl2_image" ;;
                glut) pkg_name="freeglut" ;;
                libuv) pkg_name="libuv" ;;
                libevent) pkg_name="libevent" ;;
                libonnxruntime) pkg_name="onnxruntime" ;;
                *) pkg_name="$pkg_config_name" ;;
            esac
            ;;
        debian)
            case "$pkg_config_name" in
                sdl2) pkg_name="libsdl2-dev" ;;
                SDL2_ttf) pkg_name="libsdl2-ttf-dev" ;;
                SDL2_mixer) pkg_name="libsdl2-mixer-dev" ;;
                SDL2_image) pkg_name="libsdl2-image-dev" ;;
                glut) pkg_name="freeglut3-dev" ;;
                glew) pkg_name="libglew-dev" ;;
                glfw3) pkg_name="libglfw3-dev" ;;
                libuv) pkg_name="libuv1-dev" ;;
                libevent) pkg_name="libevent-dev" ;;
                libcurl) pkg_name="libcurl4-openssl-dev" ;;
                sqlite3) pkg_name="libsqlite3-dev" ;;
                libonnxruntime) pkg_name="libonnxruntime-dev" ;;
                *) pkg_name="${pkg_config_name}-dev" ;;
            esac
            ;;
        fedora)
            case "$pkg_config_name" in
                sdl2) pkg_name="SDL2-devel" ;;
                SDL2_ttf) pkg_name="SDL2_ttf-devel" ;;
                SDL2_mixer) pkg_name="SDL2_mixer-devel" ;;
                SDL2_image) pkg_name="SDL2_image-devel" ;;
                glut) pkg_name="freeglut-devel" ;;
                libuv) pkg_name="libuv-devel" ;;
                libevent) pkg_name="libevent-devel" ;;
                libonnxruntime) pkg_name="onnxruntime-devel" ;;
                *) pkg_name="${pkg_config_name}-devel" ;;
            esac
            ;;
        arch)
            case "$pkg_config_name" in
                sdl2) pkg_name="sdl2" ;;
                SDL2_ttf) pkg_name="sdl2_ttf" ;;
                SDL2_mixer) pkg_name="sdl2_mixer" ;;
                SDL2_image) pkg_name="sdl2_image" ;;
                glut) pkg_name="freeglut" ;;
                libuv) pkg_name="libuv" ;;
                libevent) pkg_name="libevent" ;;
                libonnxruntime) pkg_name="onnxruntime" ;;
                *) pkg_name="$pkg_config_name" ;;
            esac
            ;;
        *)
            pkg_name="$pkg_config_name"
            ;;
    esac
    
    echo "$pkg_name"
}

# Check if pkg-config package is available
check_pkg_config() {
    local pkg="$1"
    
    if ! command -v pkg-config >/dev/null 2>&1; then
        # Try Homebrew's pkg-config on macOS
        if [[ "$OS" == "macos" ]] && [[ -x /opt/homebrew/bin/pkg-config ]]; then
            /opt/homebrew/bin/pkg-config --exists "$pkg" 2>/dev/null
        else
            return 1
        fi
    else
        pkg-config --exists "$pkg" 2>/dev/null
    fi
}

# Get pkg-config version
get_pkg_version() {
    local pkg="$1"
    
    if ! command -v pkg-config >/dev/null 2>&1; then
        if [[ "$OS" == "macos" ]] && [[ -x /opt/homebrew/bin/pkg-config ]]; then
            /opt/homebrew/bin/pkg-config --modversion "$pkg" 2>/dev/null || echo "unknown"
        else
            echo "unknown"
        fi
    else
        pkg-config --modversion "$pkg" 2>/dev/null || echo "unknown"
    fi
}

# Parse module.json and extract pkg_config dependencies
parse_module_json() {
    local module_json="$1"
    
    # Use Python for robust JSON parsing if available
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import json
import sys
try:
    with open('$module_json', 'r') as f:
        data = json.load(f)
        pkg_config = data.get('pkg_config', [])
        for pkg in pkg_config:
            print(pkg)
except:
    pass
" 2>/dev/null
        return
    fi
    
    # Fallback: Extract only the pkg_config array using awk
    # This is more precise than the previous grep/sed approach
    if grep -q '"pkg_config"' "$module_json"; then
        awk '
            /"pkg_config":/ {
                in_array = 1
                # Check if array is on same line
                if ($0 ~ /\[.*\]/) {
                    # Single line array
                    match($0, /\[(.*)\]/, arr)
                    split(arr[1], items, ",")
                    for (i in items) {
                        gsub(/[" \t]/, "", items[i])
                        if (items[i] != "") print items[i]
                    }
                    in_array = 0
                }
                next
            }
            in_array && /\]/ {
                in_array = 0
                next
            }
            in_array && /"[^"]*"/ {
                match($0, /"([^"]*)"/, item)
                if (item[1] != "") print item[1]
            }
        ' "$module_json"
    fi
}

# Check dependencies for a single module
check_module() {
    local module_dir="$1"
    local module_name="$(basename "$module_dir")"
    local module_json="$module_dir/module.json"
    
    if [[ ! -f "$module_json" ]]; then
        # Pure nanolang module, no dependencies
        return 0
    fi
    
    TOTAL_MODULES=$((TOTAL_MODULES + 1))
    
    # Parse pkg_config dependencies
    local deps=$(parse_module_json "$module_json")
    
    if [[ -z "$deps" ]]; then
        # No pkg_config dependencies
        return 0
    fi
    
    MODULES_WITH_DEPS=$((MODULES_WITH_DEPS + 1))
    
    local has_missing=0
    
    echo -e "${BLUE}Module: ${module_name}${NC}"
    
    for dep in $deps; do
        if check_pkg_config "$dep"; then
            local version=$(get_pkg_version "$dep")
            echo -e "  ${GREEN}✓${NC} $dep ($version)"
            AVAILABLE_DEPS=$((AVAILABLE_DEPS + 1))
        else
            local pkg_name=$(map_package "$dep")
            echo -e "  ${RED}✗${NC} $dep ${YELLOW}(missing)${NC}"
            echo -e "    Install: ${PKG_INSTALL_CMD} ${pkg_name}"
            MISSING_DEPS=$((MISSING_DEPS + 1))
            has_missing=1
            
            # Store for later installation if in install mode
            if [[ "$MODE" == "install" ]]; then
                MISSING_PACKAGES+=("$pkg_name:$dep")
            fi
        fi
    done
    
    echo ""
    return $has_missing
}

# Print header
print_header() {
    echo ""
    echo "=================================================="
    echo "  Nanolang Module Dependency Checker"
    echo "=================================================="
    echo ""
    echo "OS: $OS"
    echo "Package Manager: $PKG_MANAGER"
    echo "Modules Directory: $MODULES_DIR"
    echo ""
}

# Print summary
print_summary() {
    echo ""
    echo "=================================================="
    echo "  Summary"
    echo "=================================================="
    echo "Total modules scanned: $TOTAL_MODULES"
    echo "Modules with C dependencies: $MODULES_WITH_DEPS"
    echo -e "Available dependencies: ${GREEN}${AVAILABLE_DEPS}${NC}"
    echo -e "Missing dependencies: ${RED}${MISSING_DEPS}${NC}"
    echo ""
    
    if [[ $MISSING_DEPS -eq 0 ]]; then
        echo -e "${GREEN}✓ All module dependencies are satisfied!${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Some dependencies are missing.${NC}"
        if [[ "$MODE" == "check" ]]; then
            echo ""
            echo "To install missing dependencies:"
            echo "  make modules-install"
            echo ""
            echo "Or install manually using the commands shown above."
        fi
        return 1
    fi
}

# Interactive installation
interactive_install() {
    if [[ ${#MISSING_PACKAGES[@]} -eq 0 ]]; then
        echo -e "${GREEN}No missing packages to install!${NC}"
        return 0
    fi
    
    echo ""
    echo "=================================================="
    echo "  Interactive Installation"
    echo "=================================================="
    echo ""
    echo "The following packages will be installed:"
    echo ""
    
    # Deduplicate package names
    local unique_packages=()
    for entry in "${MISSING_PACKAGES[@]}"; do
        local pkg_name="${entry%%:*}"
        if [[ ! " ${unique_packages[@]} " =~ " ${pkg_name} " ]]; then
            unique_packages+=("$pkg_name")
        fi
    done
    
    for pkg in "${unique_packages[@]}"; do
        echo "  - $pkg"
    done
    
    echo ""
    
    # Check if we can install without sudo (macOS Homebrew)
    if [[ "$PKG_MANAGER" == "brew" ]]; then
        echo "Command: $PKG_INSTALL_CMD ${unique_packages[*]}"
    else
        echo "Command: $PKG_INSTALL_CMD ${unique_packages[*]}"
        echo ""
        echo -e "${YELLOW}Note: This will require sudo privileges.${NC}"
    fi
    
    echo ""
    read -p "Proceed with installation? (y/N) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Installing packages..."
        
        if [[ "$PKG_MANAGER" == "unknown" ]]; then
            echo -e "${RED}Error: Cannot determine package manager for automatic installation.${NC}"
            echo "Please install packages manually."
            return 1
        fi
        
        # Install packages
        if $PKG_INSTALL_CMD "${unique_packages[@]}"; then
            echo ""
            echo -e "${GREEN}✓ Installation complete!${NC}"
            echo ""
            echo "Verifying installation..."
            echo ""
            
            # Re-check dependencies
            MODE="check"
            TOTAL_MODULES=0
            MODULES_WITH_DEPS=0
            MISSING_DEPS=0
            AVAILABLE_DEPS=0
            
            for module_dir in "$MODULES_DIR"/*; do
                if [[ -d "$module_dir" ]]; then
                    check_module "$module_dir"
                fi
            done
            
            print_summary
            return 0
        else
            echo ""
            echo -e "${RED}✗ Installation failed.${NC}"
            echo "Please check error messages above and try installing manually."
            return 1
        fi
    else
        echo ""
        echo "Installation cancelled."
        return 1
    fi
}

# Main execution
main() {
    detect_os
    print_header
    
    # Check if pkg-config is installed
    if ! command -v pkg-config >/dev/null 2>&1; then
        if [[ "$OS" != "macos" ]] || [[ ! -x /opt/homebrew/bin/pkg-config ]]; then
            echo -e "${YELLOW}⚠ Warning: pkg-config not found${NC}"
            echo "pkg-config is required for module dependency detection."
            echo ""
            case "$OS" in
                macos)
                    echo "Install: brew install pkg-config"
                    ;;
                debian)
                    echo "Install: sudo apt-get install pkg-config"
                    ;;
                fedora)
                    echo "Install: sudo dnf install pkgconf-pkg-config"
                    ;;
                arch)
                    echo "Install: sudo pacman -S pkg-config"
                    ;;
            esac
            echo ""
        fi
    fi
    
    # Array to store missing packages for installation
    declare -a MISSING_PACKAGES
    
    # Scan all modules
    echo "Scanning modules..."
    echo ""
    
    for module_dir in "$MODULES_DIR"/*; do
        if [[ -d "$module_dir" ]]; then
            check_module "$module_dir"
        fi
    done
    
    # Print summary
    if ! print_summary; then
        if [[ "$MODE" == "install" ]]; then
            interactive_install
            exit $?
        else
            exit 1
        fi
    fi
}

# Run main
main
