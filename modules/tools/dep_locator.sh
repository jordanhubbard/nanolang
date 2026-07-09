#!/bin/bash
# Dependency Locator - Find C/C++ libraries on macOS and Linux
# Outputs JSON with library paths
#
# Usage: ./dep_locator.sh <library-name> [OPTIONS]
# Example: ./dep_locator.sh SDL2
#          ./dep_locator.sh openssl --header-name openssl/ssl.h --lib-name ssl

set -e

# Parse arguments
LIB_NAME=""
HEADER_NAME=""
LIB_BASENAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --header-name)
            HEADER_NAME="$2"
            shift 2
            ;;
        --lib-name)
            LIB_BASENAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <library-name> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --header-name <name>  Specific header subdirectory (default: <name>)"
            echo "  --lib-name <name>     Specific library base name (default: <name>)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 SDL2"
            echo "  $0 openssl --header-name openssl --lib-name ssl"
            echo ""
            echo "Output: JSON with include_dirs, library_dirs, and libraries"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
        *)
            if [ -z "$LIB_NAME" ]; then
                LIB_NAME="$1"
            else
                echo "Error: Multiple library names specified" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$LIB_NAME" ]; then
    echo "Error: Library name required" >&2
    echo "Usage: $0 <library-name> [OPTIONS]" >&2
    exit 1
fi

# Set defaults
HEADER_NAME="${HEADER_NAME:-$LIB_NAME}"
LIB_BASENAME="${LIB_BASENAME:-$LIB_NAME}"

# Search prefixes
PREFIXES=("/usr" "/usr/local")
if [ -d "/opt/homebrew" ]; then
    PREFIXES+=("/opt/homebrew")
fi
if [ -d "/opt/local" ]; then
    PREFIXES+=("/opt/local")
fi

# Find include directory
INCLUDE_DIR=""
for prefix in "${PREFIXES[@]}"; do
    candidate="$prefix/include/$HEADER_NAME"
    if [ -d "$candidate" ]; then
        INCLUDE_DIR="$candidate"
        break
    fi
done

# Find library directory
LIB_DIR=""
for prefix in "${PREFIXES[@]}"; do
    candidate="$prefix/lib"
    if [ -d "$candidate" ]; then
        # Check if library actually exists
        if [ -f "$candidate/lib${LIB_BASENAME}.dylib" ] || \
           [ -f "$candidate/lib${LIB_BASENAME}.so" ] || \
           [ -f "$candidate/lib${LIB_BASENAME}.a" ]; then
            LIB_DIR="$candidate"
            break
        fi
    fi
done

# Output JSON
echo "{"
echo "  \"name\": \"$LIB_NAME\","

if [ -n "$INCLUDE_DIR" ] && [ -n "$LIB_DIR" ]; then
    echo "  \"found\": true,"
    echo "  \"origin\": \"heuristic\","
else
    echo "  \"found\": false,"
    echo "  \"origin\": \"none\","
fi

echo "  \"include_dirs\": ["
if [ -n "$INCLUDE_DIR" ]; then
    echo "    \"$INCLUDE_DIR\""
fi
echo "  ],"

echo "  \"library_dirs\": ["
if [ -n "$LIB_DIR" ]; then
    echo "    \"$LIB_DIR\""
fi
echo "  ],"

echo "  \"libraries\": ["
if [ -n "$LIB_DIR" ]; then
    echo "    \"$LIB_BASENAME\""
fi
echo "  ]"
echo "}"
