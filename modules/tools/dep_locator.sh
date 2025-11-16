#!/bin/bash
# Wrapper script for dep_locator.nano
# Allows passing command-line arguments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse arguments
LIB_NAME=""
HEADER_NAME=""
LIB_BASENAME=""
NO_PKG_CONFIG=false

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
        --no-pkg-config)
            NO_PKG_CONFIG=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 <library-name> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --header-name <name>  Specific header file name (default: <name>.h)"
            echo "  --lib-name <name>     Specific library base name (default: <name>)"
            echo "  --no-pkg-config       Skip pkg-config and only use heuristic search"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 SDL2"
            echo "  $0 openssl --header-name openssl/ssl.h --lib-name ssl"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [ -z "$LIB_NAME" ]; then
                LIB_NAME="$1"
            else
                echo "Error: Multiple library names specified"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$LIB_NAME" ]; then
    echo "Error: Library name required"
    echo "Usage: $0 <library-name> [OPTIONS]"
    echo "Use --help for more information"
    exit 1
fi

# Set environment variables for nanolang program
export DEP_LOCATOR_NAME="$LIB_NAME"
if [ -n "$HEADER_NAME" ]; then
    export DEP_LOCATOR_HEADER_NAME="$HEADER_NAME"
fi
if [ -n "$LIB_BASENAME" ]; then
    export DEP_LOCATOR_LIB_NAME="$LIB_BASENAME"
fi
if [ "$NO_PKG_CONFIG" = true ]; then
    export DEP_LOCATOR_NO_PKG_CONFIG="1"
fi

# Run nanolang dependency locator
"$PROJECT_ROOT/bin/nano" "$SCRIPT_DIR/dep_locator.nano" --call main

