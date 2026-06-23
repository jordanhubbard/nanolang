#!/bin/bash
# Install a nanolang module package

set -e

PACKAGE_FILE="$1"

if [ -z "$PACKAGE_FILE" ]; then
    echo "Usage: $0 <module-package.tar.zst>"
    echo "Example: $0 sdl.nano.tar.zst"
    exit 1
fi

if [ ! -f "$PACKAGE_FILE" ]; then
    echo "Error: Package file not found: $PACKAGE_FILE"
    exit 1
fi

# Determine installation directory
if [ -n "$NANO_MODULE_PATH" ]; then
    INSTALL_DIR="$NANO_MODULE_PATH"
else
    INSTALL_DIR="$HOME/.nanolang/modules"
fi

# Create installation directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Extract module name from package filename
MODULE_NAME=$(basename "$PACKAGE_FILE" .nano.tar.zst)

echo "Installing module: $MODULE_NAME"
echo "Installation directory: $INSTALL_DIR"

# Extract package to temporary directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

tar -I zstd -xf "$PACKAGE_FILE" -C "$TEMP_DIR"

# Verify module.json exists
if [ ! -f "$TEMP_DIR/module.json" ]; then
    echo "Error: Invalid package - module.json not found"
    exit 1
fi

# Copy package to installation directory
INSTALL_PATH="$INSTALL_DIR/$MODULE_NAME.nano.tar.zst"
cp "$PACKAGE_FILE" "$INSTALL_PATH"

echo "✓ Module installed: $INSTALL_PATH"
echo ""
echo "To use this module, set NANO_MODULE_PATH:"
echo "  export NANO_MODULE_PATH=$INSTALL_DIR"
echo ""
echo "Or add to your shell profile:"
echo "  echo 'export NANO_MODULE_PATH=$INSTALL_DIR' >> ~/.bashrc"

