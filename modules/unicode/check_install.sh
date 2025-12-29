#!/bin/bash
# Check if utf8proc is properly installed (cross-platform)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "UTF8PROC Installation Checker"
echo "=============================="
echo ""

# Detect platform
OS="$(uname -s)"
case "$OS" in
    Darwin*)
        PLATFORM="macOS"
        ;;
    Linux*)
        PLATFORM="Linux"
        # Detect Linux distribution
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            DISTRO="$ID"
        else
            DISTRO="unknown"
        fi
        ;;
    *)
        PLATFORM="Unknown"
        ;;
esac

echo -e "${BLUE}Platform:${NC} $PLATFORM"
if [ "$PLATFORM" = "Linux" ]; then
    echo -e "${BLUE}Distribution:${NC} $DISTRO"
fi
echo ""

CHECKS_PASSED=0
CHECKS_TOTAL=0

# Check 1: pkg-config
echo -e "${BLUE}[1/5]${NC} Checking pkg-config..."
((CHECKS_TOTAL++))
if command -v pkg-config &> /dev/null; then
    PKG_CONFIG_VERSION=$(pkg-config --version)
    echo -e "  ${GREEN}✓${NC} Found pkg-config $PKG_CONFIG_VERSION"
    ((CHECKS_PASSED++))
else
    echo -e "  ${RED}✗${NC} pkg-config not found"
    echo -e "  ${YELLOW}Install:${NC}"
    if [ "$PLATFORM" = "macOS" ]; then
        echo "    brew install pkg-config"
    else
        echo "    sudo apt-get install pkg-config"
    fi
fi
echo ""

# Check 2: utf8proc via pkg-config
echo -e "${BLUE}[2/5]${NC} Checking utf8proc (pkg-config)..."
((CHECKS_TOTAL++))
if pkg-config --exists libutf8proc 2>/dev/null; then
    UTF8PROC_VERSION=$(pkg-config --modversion libutf8proc)
    echo -e "  ${GREEN}✓${NC} Found utf8proc $UTF8PROC_VERSION"
    ((CHECKS_PASSED++))
else
    echo -e "  ${RED}✗${NC} utf8proc not found via pkg-config"
    echo -e "  ${YELLOW}Install:${NC}"
    if [ "$PLATFORM" = "macOS" ]; then
        echo "    brew install utf8proc"
    elif [ "$DISTRO" = "ubuntu" ] || [ "$DISTRO" = "debian" ]; then
        echo "    sudo apt-get install libutf8proc-dev"
    elif [ "$DISTRO" = "fedora" ] || [ "$DISTRO" = "rhel" ] || [ "$DISTRO" = "centos" ]; then
        echo "    sudo yum install utf8proc-devel"
    elif [ "$DISTRO" = "arch" ]; then
        echo "    sudo pacman -S libutf8proc"
    else
        echo "    Install utf8proc using your package manager"
    fi
fi
echo ""

# Check 3: Header file
echo -e "${BLUE}[3/5]${NC} Checking utf8proc header..."
((CHECKS_TOTAL++))
if pkg-config --exists libutf8proc 2>/dev/null; then
    INCLUDE_DIR=$(pkg-config --variable=includedir libutf8proc)
    if [ -f "$INCLUDE_DIR/utf8proc.h" ]; then
        echo -e "  ${GREEN}✓${NC} Found header at $INCLUDE_DIR/utf8proc.h"
        ((CHECKS_PASSED++))
    else
        echo -e "  ${YELLOW}⚠${NC}  Header not at expected location"
        # Try to find it anyway
        if [ "$PLATFORM" = "macOS" ]; then
            if [ -f "/opt/homebrew/include/utf8proc.h" ]; then
                echo -e "  ${GREEN}✓${NC} Found at /opt/homebrew/include/utf8proc.h"
                ((CHECKS_PASSED++))
            elif [ -f "/usr/local/include/utf8proc.h" ]; then
                echo -e "  ${GREEN}✓${NC} Found at /usr/local/include/utf8proc.h"
                ((CHECKS_PASSED++))
            fi
        elif [ -f "/usr/include/utf8proc.h" ]; then
            echo -e "  ${GREEN}✓${NC} Found at /usr/include/utf8proc.h"
            ((CHECKS_PASSED++))
        fi
    fi
else
    echo -e "  ${YELLOW}⚠${NC}  Skipped (utf8proc not found)"
fi
echo ""

# Check 4: Library file
echo -e "${BLUE}[4/5]${NC} Checking utf8proc library..."
((CHECKS_TOTAL++))
if pkg-config --exists libutf8proc 2>/dev/null; then
    LIBDIR=$(pkg-config --variable=libdir libutf8proc)
    FOUND_LIB=false
    
    if [ "$PLATFORM" = "macOS" ]; then
        if [ -f "$LIBDIR/libutf8proc.dylib" ]; then
            echo -e "  ${GREEN}✓${NC} Found library at $LIBDIR/libutf8proc.dylib"
            FOUND_LIB=true
        fi
    else
        if [ -f "$LIBDIR/libutf8proc.so" ]; then
            echo -e "  ${GREEN}✓${NC} Found library at $LIBDIR/libutf8proc.so"
            FOUND_LIB=true
        fi
    fi
    
    if [ "$FOUND_LIB" = true ]; then
        ((CHECKS_PASSED++))
    else
        echo -e "  ${RED}✗${NC} Library not found at expected location"
    fi
else
    echo -e "  ${YELLOW}⚠${NC}  Skipped (utf8proc not found)"
fi
echo ""

# Check 5: Compiler flags
echo -e "${BLUE}[5/5]${NC} Checking compiler flags..."
((CHECKS_TOTAL++))
if pkg-config --exists libutf8proc 2>/dev/null; then
    CFLAGS=$(pkg-config --cflags libutf8proc)
    LDFLAGS=$(pkg-config --libs libutf8proc)
    echo -e "  ${GREEN}✓${NC} CFLAGS:  $CFLAGS"
    echo -e "  ${GREEN}✓${NC} LDFLAGS: $LDFLAGS"
    ((CHECKS_PASSED++))
else
    echo -e "  ${RED}✗${NC} Cannot get compiler flags"
fi
echo ""

# Summary
echo "=============================="
echo "SUMMARY: $CHECKS_PASSED/$CHECKS_TOTAL checks passed"
echo "=============================="

if [ $CHECKS_PASSED -eq $CHECKS_TOTAL ]; then
    echo -e "${GREEN}✓ All checks passed! UTF8PROC is properly installed.${NC}"
    echo ""
    echo "You can now build the Unicode module:"
    echo "  cd modules/unicode"
    echo "  ./build.sh"
    exit 0
elif [ $CHECKS_PASSED -ge 3 ]; then
    echo -e "${YELLOW}⚠ Most checks passed, but some issues detected.${NC}"
    echo "The Unicode module may still work."
    exit 0
else
    echo -e "${RED}✗ Installation incomplete or not found.${NC}"
    echo ""
    echo "Quick install:"
    if [ "$PLATFORM" = "macOS" ]; then
        echo "  brew install utf8proc"
    elif [ "$DISTRO" = "ubuntu" ] || [ "$DISTRO" = "debian" ]; then
        echo "  sudo apt-get update && sudo apt-get install libutf8proc-dev"
    elif [ "$DISTRO" = "fedora" ] || [ "$DISTRO" = "rhel" ] || [ "$DISTRO" = "centos" ]; then
        echo "  sudo yum install utf8proc-devel"
    elif [ "$DISTRO" = "arch" ]; then
        echo "  sudo pacman -S libutf8proc"
    else
        echo "  Install utf8proc using your package manager, or:"
        echo "  git clone https://github.com/JuliaStrings/utf8proc.git"
        echo "  cd utf8proc && make && sudo make install"
        if [ "$PLATFORM" = "Linux" ]; then
            echo "  sudo ldconfig"
        fi
    fi
    exit 1
fi

