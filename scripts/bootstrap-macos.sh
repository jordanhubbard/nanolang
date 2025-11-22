#!/bin/bash
# Bootstrap script for nanolang on macOS
# Installs Homebrew (if needed) and required dependencies for building examples

set -e  # Exit on error

echo "======================================"
echo "nanolang macOS Bootstrap"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is for macOS only.${NC}"
    exit 1
fi

echo "Checking dependencies..."
echo ""

# Check for Xcode Command Line Tools (required for Homebrew)
if ! xcode-select -p &> /dev/null; then
    echo -e "${YELLOW}Xcode Command Line Tools not found.${NC}"
    echo "Installing Xcode Command Line Tools..."
    xcode-select --install
    echo ""
    echo -e "${YELLOW}Please complete the Xcode Command Line Tools installation${NC}"
    echo -e "${YELLOW}and run this script again.${NC}"
    exit 0
fi

echo -e "${GREEN}✓${NC} Xcode Command Line Tools installed"

# Check for Homebrew - look in standard locations first
BREW_PATH=""
if command -v brew &> /dev/null; then
    BREW_PATH=$(command -v brew)
elif [[ $(uname -m) == 'arm64' ]] && [[ -x /opt/homebrew/bin/brew ]]; then
    # Apple Silicon - Homebrew installed but not in PATH
    BREW_PATH="/opt/homebrew/bin/brew"
    eval "$(/opt/homebrew/bin/brew shellenv)"
elif [[ -x /usr/local/bin/brew ]]; then
    # Intel Mac - Homebrew installed but not in PATH
    BREW_PATH="/usr/local/bin/brew"
    eval "$(/usr/local/bin/brew shellenv)"
fi

if [[ -z "$BREW_PATH" ]]; then
    echo -e "${YELLOW}Homebrew not found.${NC}"
    echo "Installing Homebrew..."
    echo ""
    
    # Check if running in TTY (interactive terminal)
    if [ -t 0 ]; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # For Apple Silicon Macs, add Homebrew to PATH
        if [[ $(uname -m) == 'arm64' ]]; then
            echo ""
            echo "Configuring Homebrew for Apple Silicon..."
            # Add to shell profile if not already present
            if ! grep -q '/opt/homebrew/bin/brew shellenv' ~/.zprofile 2>/dev/null; then
                echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            fi
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [[ $(uname -m) == 'x86_64' ]]; then
            # Intel Mac
            if ! grep -q '/usr/local/bin/brew shellenv' ~/.zprofile 2>/dev/null; then
                echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
            fi
            eval "$(/usr/local/bin/brew shellenv)"
        fi
        
        echo -e "${GREEN}✓${NC} Homebrew installed successfully"
    else
        echo -e "${RED}Error: Cannot install Homebrew in non-interactive mode.${NC}"
        echo ""
        echo "Please install Homebrew manually by running in your terminal:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo ""
        echo "Then run: make examples"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} Homebrew already installed at: $BREW_PATH"
fi

echo ""

# Install SDL2 (idempotent - only install if not present)
if command -v pkg-config &> /dev/null && pkg-config --exists sdl2 2>/dev/null; then
    SDL2_VERSION=$(pkg-config --modversion sdl2)
    echo -e "${GREEN}✓${NC} SDL2 already installed (version $SDL2_VERSION)"
else
    echo "Installing SDL2..."
    brew install sdl2
    echo -e "${GREEN}✓${NC} SDL2 installed successfully"
fi

echo ""

# Optional: Install OpenGL dependencies (GLFW, GLEW) for OpenGL examples
# Check if already installed first (idempotent)
GLFW_INSTALLED=false
GLEW_INSTALLED=false

if command -v pkg-config &> /dev/null; then
    if pkg-config --exists glfw3 2>/dev/null; then
        GLFW_INSTALLED=true
        echo -e "${GREEN}✓${NC} GLFW already installed"
    fi
    if pkg-config --exists glew 2>/dev/null; then
        GLEW_INSTALLED=true
        echo -e "${GREEN}✓${NC} GLEW already installed"
    fi
fi

# Only prompt if running interactively and not all deps are installed
if [ -t 0 ] && { [ "$GLFW_INSTALLED" = false ] || [ "$GLEW_INSTALLED" = false ]; }; then
    read -p "Install OpenGL dependencies (GLFW, GLEW) for OpenGL examples? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ "$GLFW_INSTALLED" = false ]; then
            echo "Installing GLFW..."
            brew install glfw
            echo -e "${GREEN}✓${NC} GLFW installed successfully"
        fi
        
        if [ "$GLEW_INSTALLED" = false ]; then
            echo "Installing GLEW..."
            brew install glew
            echo -e "${GREEN}✓${NC} GLEW installed successfully"
        fi
    fi
elif [ ! -t 0 ]; then
    # Non-interactive mode - skip if any are missing
    if [ "$GLFW_INSTALLED" = false ] || [ "$GLEW_INSTALLED" = false ]; then
        echo "Skipping OpenGL dependencies (non-interactive mode)"
        echo "To install later: brew install glfw glew"
    fi
fi

echo ""
echo "======================================"
echo "Verification"
echo "======================================"
echo ""

# Verify installations
echo "Checking installed packages..."
echo ""

if command -v pkg-config &> /dev/null && pkg-config --exists sdl2; then
    SDL2_VERSION=$(pkg-config --modversion sdl2)
    echo -e "${GREEN}✓${NC} SDL2 version: $SDL2_VERSION"
    echo "  Location: $(pkg-config --variable=prefix sdl2)"
else
    echo -e "${RED}✗${NC} SDL2 verification failed"
    exit 1
fi

if command -v pkg-config &> /dev/null && pkg-config --exists glfw3; then
    GLFW_VERSION=$(pkg-config --modversion glfw3)
    echo -e "${GREEN}✓${NC} GLFW version: $GLFW_VERSION"
fi

if command -v pkg-config &> /dev/null && pkg-config --exists glew; then
    GLEW_VERSION=$(pkg-config --modversion glew)
    echo -e "${GREEN}✓${NC} GLEW version: $GLEW_VERSION"
fi

echo ""
echo "======================================"
echo "Bootstrap Complete!"
echo "======================================"
echo ""
echo "You can now build nanolang examples:"
echo "  make                  # Build compiler and interpreter"
echo "  make examples         # Build all examples"
echo "  make -C examples sdl  # Build SDL examples only"
echo ""
echo "Run examples:"
echo "  ./bin/checkers_sdl"
echo "  ./bin/boids_sdl"
echo "  ./bin/particles_sdl"
echo ""
