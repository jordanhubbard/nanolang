#!/usr/bin/env bash
#
# Nanolang Watch Mode
#
# Monitors nanolang source files for changes and automatically recompiles.
# Useful for rapid development iteration.
#
# Usage:
#   ./scripts/watch.sh [file.nano]           # Watch and compile specific file
#   ./scripts/watch.sh --test                # Watch and run tests
#   ./scripts/watch.sh --bootstrap           # Watch and rebuild compiler
#   ./scripts/watch.sh --examples            # Watch and build examples
#   ./scripts/watch.sh --interpreter file    # Watch and run in interpreter

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

MODE="compile"
TARGET_FILE=""
WATCH_PATHS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MODE="test"
            WATCH_PATHS=("tests/" "src/" "src_nano/")
            shift
            ;;
        --bootstrap)
            MODE="bootstrap"
            WATCH_PATHS=("src/" "src_nano/")
            shift
            ;;
        --examples)
            MODE="examples"
            WATCH_PATHS=("examples/" "src/" "std/")
            shift
            ;;
        --interpreter)
            MODE="interpreter"
            TARGET_FILE="$2"
            WATCH_PATHS=("$TARGET_FILE")
            shift 2
            ;;
        --help|-h)
            echo "Nanolang Watch Mode"
            echo ""
            echo "Usage:"
            echo "  $0 [file.nano]           # Watch and compile specific file"
            echo "  $0 --test                # Watch and run tests"
            echo "  $0 --bootstrap           # Watch and rebuild compiler"
            echo "  $0 --examples            # Watch and build examples"
            echo "  $0 --interpreter file    # Watch and run in interpreter"
            echo ""
            exit 0
            ;;
        *.nano)
            TARGET_FILE="$1"
            WATCH_PATHS=("$TARGET_FILE" "std/")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Default to watching current directory if no paths specified
if [ ${#WATCH_PATHS[@]} -eq 0 ]; then
    WATCH_PATHS=("src/" "src_nano/" "examples/" "tests/" "std/")
fi

cd "$PROJECT_ROOT"

# Check if fswatch is available (preferred)
if command -v fswatch > /dev/null 2>&1; then
    WATCHER="fswatch"
    echo -e "${GREEN}Using fswatch for file monitoring${NC}"
elif command -v inotifywait > /dev/null 2>&1; then
    WATCHER="inotifywait"
    echo -e "${GREEN}Using inotifywait for file monitoring${NC}"
else
    echo -e "${RED}Error: No file watcher found${NC}"
    echo ""
    echo "Please install one of the following:"
    echo "  macOS:   brew install fswatch"
    echo "  Ubuntu:  sudo apt-get install inotify-tools"
    echo "  Fedora:  sudo dnf install inotify-tools"
    exit 1
fi

# Function to run the appropriate action
run_action() {
    clear
    echo -e "${CYAN}======================================${NC}"
    echo -e "${CYAN}Nanolang Watch Mode - $(date +%H:%M:%S)${NC}"
    echo -e "${CYAN}======================================${NC}"
    echo ""
    
    case $MODE in
        compile)
            if [ -n "$TARGET_FILE" ]; then
                echo -e "${YELLOW}Compiling $TARGET_FILE...${NC}"
                if ./bin/nanoc "$TARGET_FILE" -o "${TARGET_FILE%.nano}"; then
                    echo ""
                    echo -e "${GREEN}✅ Compilation successful!${NC}"
                else
                    echo ""
                    echo -e "${RED}❌ Compilation failed${NC}"
                fi
            else
                echo -e "${YELLOW}Rebuilding project...${NC}"
                if make build > /dev/null 2>&1; then
                    echo -e "${GREEN}✅ Build successful!${NC}"
                else
                    echo -e "${RED}❌ Build failed${NC}"
                fi
            fi
            ;;
        test)
            echo -e "${YELLOW}Running tests...${NC}"
            if make test; then
                echo ""
                echo -e "${GREEN}✅ All tests passed!${NC}"
            else
                echo ""
                echo -e "${RED}❌ Tests failed${NC}"
            fi
            ;;
        bootstrap)
            echo -e "${YELLOW}Rebuilding compiler...${NC}"
            if make clean > /dev/null 2>&1 && make build; then
                echo ""
                echo -e "${GREEN}✅ Bootstrap successful!${NC}"
            else
                echo ""
                echo -e "${RED}❌ Bootstrap failed${NC}"
            fi
            ;;
        examples)
            echo -e "${YELLOW}Building examples...${NC}"
            if make examples; then
                echo ""
                echo -e "${GREEN}✅ Examples built successfully!${NC}"
            else
                echo ""
                echo -e "${RED}❌ Examples build failed${NC}"
            fi
            ;;
        interpreter)
            if [ -n "$TARGET_FILE" ]; then
                echo -e "${YELLOW}Running $TARGET_FILE in interpreter...${NC}"
                echo ""
                if ./bin/nano "$TARGET_FILE"; then
                    echo ""
                    echo -e "${GREEN}✅ Execution successful!${NC}"
                else
                    echo ""
                    echo -e "${RED}❌ Execution failed (exit code: $?)${NC}"
                fi
            fi
            ;;
    esac
    
    echo ""
    echo -e "${CYAN}Watching for changes...${NC}"
    echo -e "${CYAN}(Press Ctrl+C to stop)${NC}"
}

# Run once initially
run_action

# Watch for changes
if [ "$WATCHER" = "fswatch" ]; then
    # fswatch command (macOS, cross-platform)
    fswatch -r "${WATCH_PATHS[@]}" --event Updated --event Created --event Removed \
        --exclude ".*\.o$" --exclude ".*\.a$" --exclude ".*\.dylib$" --exclude ".*\.so$" \
        --exclude ".*/\.git/.*" --exclude ".*/obj/.*" --exclude ".*/bin/.*" \
        --exclude ".*~$" --exclude ".*\.swp$" | while read -r file; do
        
        # Skip if file doesn't end in .nano, .c, or .h
        if [[ ! "$file" =~ \.(nano|c|h)$ ]]; then
            continue
        fi
        
        echo ""
        echo -e "${YELLOW}Change detected: $file${NC}"
        sleep 0.5  # Debounce
        run_action
    done
else
    # inotifywait command (Linux)
    while true; do
        inotifywait -r -q -e modify,create,delete --exclude '\.(o|a|so|dylib)$' "${WATCH_PATHS[@]}" && {
            sleep 0.5  # Debounce
            run_action
        }
    done
fi

