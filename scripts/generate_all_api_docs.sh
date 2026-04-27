#!/bin/bash
# Generate API documentation for all major modules
#
# NOTE: This script uses the NanoLang-based API doc generator.
# It compiles scripts/generate_module_api_docs.nano and runs it per module.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
API_DOCS_DIR="$PROJECT_ROOT/userguide/api_reference"

cd "$PROJECT_ROOT"
export NANO_MODULE_PATH="$PROJECT_ROOT/modules"

echo "Generating API documentation for all modules..."
echo ""

# Create output directory
mkdir -p "$API_DOCS_DIR"

# Compile the NanoLang API doc generator
GEN_TOOL="$PROJECT_ROOT/build/userguide/generate_module_api_docs"
mkdir -p "$PROJECT_ROOT/build/userguide"
perl -e 'alarm 180; exec @ARGV' ./bin/nanoc scripts/generate_module_api_docs.nano -o "$GEN_TOOL"

# Function to generate API docs for a module
generate_api_doc() {
    local module_path="$1"
    local output_name="$2"
    
    if [ ! -f "$module_path" ]; then
        echo "  ⚠️  Skipping $output_name (file not found: $module_path)"
        return
    fi
    
    local md_path="$API_DOCS_DIR/${output_name}.md"
    echo "  → $output_name..."
    if perl -e 'alarm 180; exec @ARGV' "$GEN_TOOL" "$module_path" "$md_path"; then
        echo "    ✓ Generated $md_path"
    else
        echo "    ⚠️  Failed to generate $md_path (timeout or error)"
    fi
}

# Stdlib modules
echo "📚 Standard Library:"
generate_api_doc "stdlib/log.nano" "log"
generate_api_doc "stdlib/coverage.nano" "coverage"
generate_api_doc "modules/std/collections/stringbuilder.nano" "StringBuilder"
generate_api_doc "std/regex/regex.nano" "regex"

# SDL family
echo ""
echo "🎮 SDL Family:"
generate_api_doc "modules/sdl/sdl.nano" "sdl"
generate_api_doc "modules/sdl_image/sdl_image.nano" "sdl_image"
generate_api_doc "modules/sdl_mixer/sdl_mixer.nano" "sdl_mixer"
generate_api_doc "modules/sdl_ttf/sdl_ttf.nano" "sdl_ttf"

# Terminal
echo ""
echo "💻 Terminal:"
generate_api_doc "modules/ncurses/ncurses.nano" "ncurses"

# Network
echo ""
echo "🌐 Network:"
generate_api_doc "modules/curl/curl.nano" "curl"
generate_api_doc "modules/http_server/http_server.nano" "http_server"
generate_api_doc "modules/uv/uv.nano" "uv"

# Data
echo ""
echo "💾 Data:"
generate_api_doc "modules/sqlite/sqlite.nano" "sqlite"

# Graphics
echo ""
echo "🎨 Graphics:"
generate_api_doc "modules/opengl/opengl.nano" "opengl"
generate_api_doc "modules/glew/glew.nano" "glew"
generate_api_doc "modules/glfw/glfw.nano" "glfw"
generate_api_doc "modules/glut/glut.nano" "glut"

# Physics
echo ""
echo "⚛️  Physics:"
generate_api_doc "modules/bullet/bullet.nano" "bullet"

# Utilities
echo ""
echo "🔧 Utilities:"
generate_api_doc "modules/filesystem/filesystem.nano" "filesystem"
generate_api_doc "modules/preferences/preferences.nano" "preferences"
generate_api_doc "modules/event/event.nano" "event"
generate_api_doc "modules/vector2d/vector2d.nano" "vector2d"
generate_api_doc "modules/proptest/proptest.nano" "proptest"

echo ""
echo "✅ API documentation generation complete!"
echo "   Output: $API_DOCS_DIR/"
