#!/usr/bin/env python3
"""
Extract individual icon images from a grid layout.
Usage: python3 extract_icons.py <input_image_path>
"""

import sys
import subprocess
import os

# Icon definitions based on the grid layout (row, col, filename)
icons = [
    # Row 1 (5 icons)
    (0, 0, "sdl_asteroids"),
    (0, 1, "sdl_checkers"),
    (0, 2, "sdl_pong"),
    (0, 3, "sdl_snake"),
    (0, 4, "sdl_mouse_click"),
    
    # Row 2 (6 icons)
    (1, 0, "sdl_fire"),
    (1, 1, "sdl_particles"),
    (1, 2, "sdl_starfield"),
    (1, 3, "sdl_boids"),
    (1, 4, "sdl_falling_sand"),
    (1, 5, "sdl_raytracer"),
    
    # Row 3 (5 icons)
    (2, 0, "sdl_drawing_primitives"),
    (2, 1, "sdl_glass_sphere"),  # This appears to be a duplicate particles in the image
    (2, 2, "sdl_terrain_explorer"),  # This appears labeled as xtarfield
    (2, 3, "sdl_texture_demo"),
    (2, 4, "sdl_ui_widgets_extended"),
    
    # Row 4 (5 icons)  
    (3, 0, "sdl_audio_wav"),
    (3, 1, "sdl_nanoamp"),
    (3, 2, "sdl_mod_visualizer"),
    (3, 3, "sdl_tracker_shell"),
    (3, 4, "sdl_audio_player"),
    
    # Row 5 (3 icons)
    (4, 0, "sdl_nanoviz"),  # This appears labeled as sdl_audio_wav in image
    (4, 1, "sdl_example_launcher"),  # This appears labeled as sdl_nanoamp in image
    (4, 2, "sdl_audio_wav_alt"),  # This appears labeled as sdl_mo_vcanog in image
]

def extract_icons(input_path):
    """Extract icons using macOS sips command."""
    
    # Get image dimensions
    result = subprocess.run(
        ['sips', '-g', 'pixelWidth', '-g', 'pixelHeight', input_path],
        capture_output=True, text=True
    )
    
    lines = result.stdout.strip().split('\n')
    width = int([l for l in lines if 'pixelWidth' in l][0].split(':')[1].strip())
    height = int([l for l in lines if 'pixelHeight' in l][0].split(':')[1].strip())
    
    print(f"Image size: {width}x{height}")
    
    # Estimate grid parameters
    # Looking at the image: icons appear to be ~160-180px wide with labels
    # Let's estimate based on 5 columns for first row
    icon_width = 180  # Approximate width including padding
    icon_height = 220  # Approximate height including label
    icon_size = 80  # Actual icon size we want
    
    # Starting offsets (left margin and top margin)
    start_x = 20
    start_y = 30
    
    output_dir = "examples/icons"
    os.makedirs(output_dir, exist_ok=True)
    
    for row, col, name in icons:
        # Calculate position
        x = start_x + (col * icon_width)
        y = start_y + (row * icon_height)
        
        output_file = f"{output_dir}/{name}.png"
        
        # Extract using sips cropOffset
        cmd = [
            'sips',
            '-c', str(icon_size), str(icon_size),  # crop to size
            '--cropOffset', str(y), str(x),  # y, x offset
            input_path,
            '--out', output_file
        ]
        
        print(f"Extracting {name} from position ({x}, {y})...")
        subprocess.run(cmd, capture_output=True)
    
    print(f"\n✓ Extracted {len(icons)} icons to {output_dir}/")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_icons.py <input_image_path>")
        print("\nPlease save the grid image and provide its path.")
        sys.exit(1)
    
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    extract_icons(input_path)

