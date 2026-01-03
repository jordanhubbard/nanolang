#!/bin/bash
# Precise icon extraction - measures based on visible grid
# Usage: ./extract_icons_precise.sh <path_to_grid_image>

INPUT="$1"
OUTPUT_DIR="examples/icons"

if [ -z "$INPUT" ]; then
    echo "Usage: $0 <path_to_grid_image>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Grid measurements (adjust these after checking the actual image)
ICON_SIZE=80
COL_SPACING=200  # Distance between icon centers horizontally
ROW_SPACING=220  # Distance between icon centers vertically
START_X=50       # X offset to first icon center
START_Y=90       # Y offset to first icon center
HALF=$((ICON_SIZE / 2))

# Function to extract an icon
extract_icon() {
    local name=$1
    local row=$2
    local col=$3
    
    local center_x=$((START_X + col * COL_SPACING))
    local center_y=$((START_Y + row * ROW_SPACING))
    local crop_x=$((center_x - HALF))
    local crop_y=$((center_y - HALF))
    
    echo "Extracting $name (row $row, col $col) from offset ($crop_x, $crop_y)..."
    
    # Create temp cropped file
    sips -c $ICON_SIZE $ICON_SIZE --cropOffset $crop_y $crop_x "$INPUT" --out "$OUTPUT_DIR/${name}.png" 2>/dev/null
}

echo "Extracting icons to $OUTPUT_DIR/"
echo ""

# Row 0 (5 icons)
extract_icon "sdl_asteroids" 0 0
extract_icon "sdl_checkers" 0 1
extract_icon "sdl_pong" 0 2
extract_icon "sdl_snake" 0 3
extract_icon "sdl_mouse_click" 0 4

# Row 1 (6 icons)
extract_icon "sdl_fire" 1 0
extract_icon "sdl_particles" 1 1
extract_icon "sdl_starfield" 1 2
extract_icon "sdl_boids" 1 3
extract_icon "sdl_falling_sand" 1 4
extract_icon "sdl_raytracer" 1 5

# Row 2 (5 icons)
extract_icon "sdl_drawing_primitives" 2 0
extract_icon "sdl_glass_sphere" 2 1
extract_icon "sdl_terrain_explorer" 2 2
extract_icon "sdl_texture_demo" 2 3
extract_icon "sdl_ui_widgets_extended" 2 4

# Row 3 (5 icons)
extract_icon "sdl_audio_wav" 3 0
extract_icon "sdl_nanoamp" 3 1
extract_icon "sdl_mod_visualizer" 3 2
extract_icon "sdl_tracker_shell" 3 3
extract_icon "sdl_audio_player" 3 4

# Row 4 (3 icons)
extract_icon "sdl_nanoviz" 4 0
extract_icon "sdl_example_launcher" 4 1

echo ""
echo "✓ Done! Check $OUTPUT_DIR/ for extracted icons"
echo ""
echo "If icons are misaligned, adjust START_X, START_Y, COL_SPACING, or ROW_SPACING"
echo "in this script and re-run."

