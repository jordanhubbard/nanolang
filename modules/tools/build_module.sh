#!/bin/bash
# Build a nanolang module from source and create a package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODULE_DIR="$1"

if [ -z "$MODULE_DIR" ]; then
    echo "Usage: $0 <module-directory>"
    echo "Example: $0 ../sdl"
    exit 1
fi

if [ ! -d "$MODULE_DIR" ]; then
    echo "Error: Module directory not found: $MODULE_DIR"
    exit 1
fi

MODULE_NAME=$(basename "$MODULE_DIR")
MODULE_JSON="$MODULE_DIR/module.json"

if [ ! -f "$MODULE_JSON" ]; then
    echo "Error: module.json not found in $MODULE_DIR"
    exit 1
fi

echo "Building module: $MODULE_NAME"

# Parse module.json to get dependencies
if command -v jq >/dev/null 2>&1; then
    # Check system dependencies
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    if [ "$OS" = "darwin" ]; then
        BREW_PACKAGES=$(jq -r '.dependencies.system.macos.brew[]? // empty' "$MODULE_JSON" 2>/dev/null || echo "")
        if [ -n "$BREW_PACKAGES" ]; then
            echo "Checking macOS dependencies..."
            for pkg in $BREW_PACKAGES; do
                if ! brew list "$pkg" >/dev/null 2>&1; then
                    echo "Warning: $pkg not installed. Install with: brew install $pkg"
                    read -p "Install now? (y/n) " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        brew install "$pkg"
                    fi
                fi
            done
        fi
    elif [ "$OS" = "linux" ]; then
        APT_PACKAGES=$(jq -r '.dependencies.system.linux.apt[]? // empty' "$MODULE_JSON" 2>/dev/null || echo "")
        if [ -n "$APT_PACKAGES" ]; then
            echo "Checking Linux dependencies..."
            for pkg in $APT_PACKAGES; do
                if ! dpkg -l "$pkg" >/dev/null 2>&1; then
                    echo "Warning: $pkg not installed. Install with: sudo apt install $pkg"
                    read -p "Install now? (y/n) " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        sudo apt install "$pkg"
                    fi
                fi
            done
        fi
    fi
else
    echo "Warning: jq not found. Skipping dependency checking."
fi

# Get source file from module.json or default
SOURCE_FILE=$(jq -r '.source_file // "'"$MODULE_NAME"'.nano"' "$MODULE_JSON" 2>/dev/null || echo "$MODULE_NAME.nano")
SOURCE_PATH="$MODULE_DIR/$SOURCE_FILE"

if [ ! -f "$SOURCE_PATH" ]; then
    echo "Error: Source file not found: $SOURCE_PATH"
    exit 1
fi

# Create temporary build directory
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "Building module from: $SOURCE_PATH"

# Compile module if it has implementation (not FFI-only)
# Check if module is FFI-only by looking for "extern fn" without regular "fn"
if grep -q "^extern fn" "$SOURCE_PATH" && ! grep -q "^fn " "$SOURCE_PATH"; then
    echo "Module is FFI-only, skipping compilation"
    HAS_IMPLEMENTATION=false
else
    echo "Compiling module..."
    HAS_IMPLEMENTATION=true
    
    # Try to discover compilation flags using dep_locator
    # First, check if module.json specifies a library name to locate
    LIBRARY_NAME=$(jq -r '.compilation.library_name // empty' "$MODULE_JSON" 2>/dev/null || echo "")
    
    if [ -n "$LIBRARY_NAME" ] && [ -f "$SCRIPT_DIR/dep_locator.sh" ]; then
        echo "Discovering compilation flags for library: $LIBRARY_NAME"
        LOCATION_OUTPUT=$("$SCRIPT_DIR/dep_locator.sh" "$LIBRARY_NAME" 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$LOCATION_OUTPUT" ]; then
            # Parse JSON output using jq
            INCLUDE_PATHS=$(echo "$LOCATION_OUTPUT" | jq -r '.include_dirs[]? // empty' 2>/dev/null | tr '\n' ' ')
            LIB_PATHS=$(echo "$LOCATION_OUTPUT" | jq -r '.library_dirs[]? // empty' 2>/dev/null | tr '\n' ' ')
            LIBRARIES=$(echo "$LOCATION_OUTPUT" | jq -r '.libraries[]? // empty' 2>/dev/null | tr '\n' ' ')
            
            if [ -n "$INCLUDE_PATHS" ] || [ -n "$LIB_PATHS" ]; then
                echo "✓ Discovered compilation flags via dependency locator"
            fi
        fi
    fi
    
    # Fall back to module.json if discovery didn't work or wasn't attempted
    if [ -z "$INCLUDE_PATHS" ] && [ -z "$LIB_PATHS" ]; then
        if [ "$OS" = "darwin" ]; then
            INCLUDE_PATHS=$(jq -r '.compilation.include_paths.macos[]? // empty' "$MODULE_JSON" 2>/dev/null || echo "")
            LIB_PATHS=$(jq -r '.compilation.library_paths.macos[]? // empty' "$MODULE_JSON" 2>/dev/null || echo "")
        else
            INCLUDE_PATHS=$(jq -r '.compilation.include_paths.linux[]? // empty' "$MODULE_JSON" 2>/dev/null || echo "")
            LIB_PATHS=$(jq -r '.compilation.library_paths.linux[]? // empty' "$MODULE_JSON" 2>/dev/null || echo "")
        fi
        
        LIBRARIES=$(jq -r '.compilation.libraries[]? // empty' "$MODULE_JSON" 2>/dev/null || echo "")
        
        if [ -n "$INCLUDE_PATHS" ] || [ -n "$LIB_PATHS" ]; then
            echo "Using compilation flags from module.json"
        fi
    fi
    
    # Build include flags
    INCLUDE_FLAGS="-Isrc"
    if [ -n "$INCLUDE_PATHS" ]; then
        for path in $INCLUDE_PATHS; do
            # Skip empty paths
            if [ -n "$path" ] && [ "$path" != "null" ]; then
                INCLUDE_FLAGS="$INCLUDE_FLAGS -I$path"
            fi
        done
    fi
    
    # Build library path flags
    LIB_PATH_FLAGS=""
    if [ -n "$LIB_PATHS" ]; then
        for path in $LIB_PATHS; do
            # Skip empty paths
            if [ -n "$path" ] && [ "$path" != "null" ]; then
                LIB_PATH_FLAGS="$LIB_PATH_FLAGS -L$path"
            fi
        done
    fi
    
    # Build library flags
    LIB_FLAGS=""
    if [ -n "$LIBRARIES" ]; then
        for lib in $LIBRARIES; do
            # Skip empty libs
            if [ -n "$lib" ] && [ "$lib" != "null" ]; then
                LIB_FLAGS="$LIB_FLAGS -l$lib"
            fi
        done
    fi
    
    # Compile module to object file
    OBJ_FILE="$BUILD_DIR/$MODULE_NAME.o"
    "$PROJECT_ROOT/bin/nanoc" "$SOURCE_PATH" -o "$OBJ_FILE" $INCLUDE_FLAGS $LIB_PATH_FLAGS $LIB_FLAGS --keep-c 2>&1 | grep -v "Shadow tests" || true
    
    if [ ! -f "$OBJ_FILE" ]; then
        echo "Error: Failed to compile module"
        exit 1
    fi
fi

# Copy files to build directory
cp "$MODULE_JSON" "$BUILD_DIR/"
cp "$SOURCE_PATH" "$BUILD_DIR/"

# Create build.json
BUILD_JSON="$BUILD_DIR/build.json"
cat > "$BUILD_JSON" <<EOF
{
  "source_file": "$SOURCE_FILE",
  "has_implementation": $HAS_IMPLEMENTATION,
  "is_ffi_only": $([ "$HAS_IMPLEMENTATION" = "false" ] && echo "true" || echo "false"),
  "compiler_version": "$("$PROJECT_ROOT/bin/nanoc" --version 2>&1 | head -1 || echo "unknown")",
  "build_date": "$(date -u +%Y-%m-%d)",
  "platform": "$(uname -sm | tr ' ' '-')"
}
EOF

# Create package archive
PACKAGE_NAME="$MODULE_NAME.nano.tar.zst"
PACKAGE_PATH="$MODULE_DIR/$PACKAGE_NAME"

echo "Creating package: $PACKAGE_NAME"

cd "$BUILD_DIR"
if command -v tar >/dev/null 2>&1 && tar --help 2>&1 | grep -q "zstd"; then
    tar -I zstd -cf "$PACKAGE_PATH" * 2>/dev/null || {
        echo "Error: Failed to create package"
        exit 1
    }
elif command -v tar >/dev/null 2>&1; then
    # Fallback: use gzip if zstd not available
    echo "Warning: zstd not available, using gzip compression"
    PACKAGE_NAME="$MODULE_NAME.nano.tar.gz"
    PACKAGE_PATH="$MODULE_DIR/$PACKAGE_NAME"
    tar -czf "$PACKAGE_PATH" * 2>/dev/null || {
        echo "Error: Failed to create package"
        exit 1
    }
else
    echo "Error: tar not found"
    exit 1
fi

echo "✓ Module package created: $PACKAGE_PATH"
echo "  Install with: ./install_module.sh $PACKAGE_PATH"

