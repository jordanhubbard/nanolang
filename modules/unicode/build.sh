#!/bin/bash
# Cross-platform build script for Unicode module
# Works on macOS and Linux

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_NAME="unicode"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Building Unicode Module (Cross-Platform)"
echo "=========================================="
echo ""

# Detect platform
OS="$(uname -s)"
case "$OS" in
    Darwin*)
        PLATFORM="macOS"
        LIB_EXT="dylib"
        ;;
    Linux*)
        PLATFORM="Linux"
        LIB_EXT="so"
        ;;
    *)
        PLATFORM="Unknown"
        LIB_EXT="so"
        ;;
esac

echo "Platform: $PLATFORM"
echo ""

# Check for pkg-config
if ! command -v pkg-config &> /dev/null; then
    echo -e "${RED}Error: pkg-config not found${NC}"
    echo ""
    if [ "$PLATFORM" = "macOS" ]; then
        echo "Install with: brew install pkg-config"
    else
        echo "Install with: sudo apt-get install pkg-config"
    fi
    exit 1
fi

# Check for utf8proc
echo "Checking for utf8proc..."
if ! pkg-config --exists libutf8proc; then
    echo -e "${RED}Error: utf8proc not found${NC}"
    echo ""
    echo "Installation instructions:"
    if [ "$PLATFORM" = "macOS" ]; then
        echo "  brew install utf8proc"
    elif [ "$PLATFORM" = "Linux" ]; then
        if command -v apt-get &> /dev/null; then
            echo "  sudo apt-get install libutf8proc-dev"
        elif command -v yum &> /dev/null; then
            echo "  sudo yum install utf8proc-devel"
        elif command -v dnf &> /dev/null; then
            echo "  sudo dnf install utf8proc-devel"
        elif command -v pacman &> /dev/null; then
            echo "  sudo pacman -S libutf8proc"
        else
            echo "  Install utf8proc using your package manager"
        fi
    fi
    echo ""
    echo "Or build from source:"
    echo "  git clone https://github.com/JuliaStrings/utf8proc.git"
    echo "  cd utf8proc && make && sudo make install"
    if [ "$PLATFORM" = "Linux" ]; then
        echo "  sudo ldconfig"
    fi
    exit 1
fi

# Get utf8proc version
UTF8PROC_VERSION=$(pkg-config --modversion libutf8proc)
echo -e "${GREEN}✓${NC} Found utf8proc $UTF8PROC_VERSION"

# Get compiler flags
CFLAGS=$(pkg-config --cflags libutf8proc)
LDFLAGS=$(pkg-config --libs libutf8proc)

echo "  CFLAGS:  $CFLAGS"
echo "  LDFLAGS: $LDFLAGS"
echo ""

# Compile FFI bindings
echo "Compiling unicode_ffi.c..."
gcc -std=c99 -fPIC -Wall -Wextra -O2 \
    $CFLAGS \
    -c "$SCRIPT_DIR/unicode_ffi.c" \
    -o "$SCRIPT_DIR/unicode_ffi.o"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Compilation successful"
    echo ""
    
    # Show object file info
    ls -lh "$SCRIPT_DIR/unicode_ffi.o"
    echo ""
    
    echo -e "${GREEN}Build complete!${NC}"
    echo ""
    echo "To use in your NanoLang program:"
    echo "  1. Import: import \"modules/unicode/unicode.nano\" as Unicode"
    echo "  2. Compile with FFI:"
    echo "     nanoc your_program.nano -o output"
    echo "     gcc -o output output.c unicode_ffi.o $LDFLAGS"
    echo ""
    echo "To test:"
    echo "  cd ../.."
    echo "  make test-unicode  # (if Makefile target exists)"
    echo "  # Or manually:"
    echo "  nanoc examples/unicode_demo.nano && link manually"
else
    echo -e "${RED}✗${NC} Compilation failed"
    exit 1
fi

# Platform-specific notes
echo ""
echo "Platform-specific notes:"
if [ "$PLATFORM" = "macOS" ]; then
    echo "  • Libraries use .dylib extension"
    echo "  • Homebrew path: /opt/homebrew (Apple Silicon) or /usr/local (Intel)"
    echo "  • Check with: otool -L your_binary"
elif [ "$PLATFORM" = "Linux" ]; then
    echo "  • Libraries use .so extension"
    echo "  • After installing utf8proc, run: sudo ldconfig"
    echo "  • Check with: ldd your_binary"
    echo "  • Library path: /usr/lib or /usr/lib/x86_64-linux-gnu"
fi

