#!/bin/bash
# Manual icon extraction using sips with measured offsets
# Usage: ./extract_icons_manual.sh <path_to_grid_image>

INPUT="$1"
OUTPUT_DIR="examples/icons"

if [ -z "$INPUT" ]; then
    echo "Usage: $0 <path_to_grid_image>"
    echo "Example: $0 ~/Downloads/icons_grid.png"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "Error: File not found: $INPUT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Get image dimensions
echo "Analyzing image dimensions..."
sips -g pixelWidth -g pixelHeight "$INPUT"

echo ""
echo "Based on the grid layout, icons appear to be arranged as:"
echo "Row 1: 5 icons (asteroids, checkers, pong, snake, mouse_click)"
echo "Row 2: 6 icons (fire, particles, starfield, boids, falling_sand, raytracer)"
echo "Row 3: 5 icons (drawing_primitives, glass_sphere, terrain_explorer, texture_demo, ui_widgets)"
echo "Row 4: 5 icons (audio_wav, nanoamp, mod_visualizer, tracker_shell, audio_player)"
echo "Row 5: 3 icons (nanoviz, example_launcher, ...)"
echo ""
echo "MANUAL EXTRACTION NEEDED:"
echo "Please use an image editor to measure the exact pixel positions,"
echo "or adjust the offsets below and re-run."
echo ""
echo "For now, you can manually extract with commands like:"
echo "  sips -c 80 80 --cropOffset Y X '$INPUT' --out '$OUTPUT_DIR/sdl_asteroids.png'"
echo ""
echo "Suggested positions (adjust as needed):"
echo "  sdl_asteroids:    X=25,  Y=35"
echo "  sdl_checkers:     X=225, Y=35"
echo "  sdl_pong:         X=435, Y=35"
echo "  sdl_snake:        X=645, Y=35"
echo "  sdl_mouse_click:  X=855, Y=35"

