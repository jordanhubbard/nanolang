# OpenGL Textured Rotating Teapot

A 3D rotating teapot demo showcasing OpenGL integration with nanolang.

## Features

- **3D Geometry**: Procedurally generated teapot with body, spout, and handle
- **Rotation**: Smooth animation on multiple axes
- **Dynamic Textures**: 5 different procedural texture patterns
- **Interactive**: Press SPACE to cycle through texture styles
- **Zero Dependencies**: All geometry and textures generated procedurally

## Installation

### macOS
```bash
brew install glfw glew
```

### Linux
```bash
sudo apt install libglfw3-dev libglew-dev
```

## Build & Run

```bash
# From examples/ directory:
make opengl-teapot

# Or from project root:
NANO_MODULE_PATH=modules ./bin/nanoc examples/opengl_teapot/teapot.nano -o bin/opengl_teapot

# Run:
./bin/opengl_teapot
```

## Controls

- **SPACE** - Cycle through texture patterns
- **ESC** - Exit

## Texture Patterns

1. **Solid** - Pure color
2. **Inverted** - Color inversion
3. **Shifted** - Hue shift (RGB → GBR)
4. **Grayscale** - Average of RGB
5. **Complementary** - Partial color complement

## Technical Details

- Uses GLFW for window management
- Uses GLEW for OpenGL function loading
- Legacy OpenGL (immediate mode) for simplicity
- Depth testing enabled for proper 3D rendering
- Orthographic projection
- ~12 latitude bands × 16 longitude bands for sphere approximation

## Code Structure

```
examples/opengl_teapot/
├── teapot.nano  - Main demo code
└── README.md    - This file
```

All geometry and colors are generated procedurally - no external asset files needed!

