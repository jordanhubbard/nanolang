# GLEW Module for nanolang

OpenGL Extension Wrangler Library - manages OpenGL function loading and extension queries.

## Installation

### macOS
```bash
brew install glew
```

### Linux (Ubuntu/Debian)
```bash
sudo apt install libglew-dev
```

### Linux (Fedora/RHEL)
```bash
sudo dnf install glew-devel
```

## Why GLEW?

Modern OpenGL uses function pointers that must be loaded at runtime. GLEW handles this automatically:

1. **Function Loading**: Loads all OpenGL functions for your platform
2. **Extension Support**: Check which extensions are available
3. **Version Detection**: Determine OpenGL version support

## Usage with GLFW

```nano
import "modules/glfw/glfw.nano"
import "modules/glew/glew.nano"

# OpenGL constants
let GL_COLOR_BUFFER_BIT: int = 0x00004000
let GLEW_OK: int = 0

fn main() -> int {
    # 1. Initialize GLFW
    if (== (glfwInit) 0) {
        (println "Failed to initialize GLFW")
        return 1
    }
    
    # 2. Create window and OpenGL context
    let window: int = (glfwCreateWindow 800 600 "OpenGL Window" 0 0)
    if (== window 0) {
        (glfwTerminate)
        return 1
    }
    
    # 3. Make context current
    (glfwMakeContextCurrent window)
    
    # 4. Initialize GLEW (AFTER making context current!)
    let glew_status: int = (glfwInit)
    if (!= glew_status GLEW_OK) {
        (println "Failed to initialize GLEW")
        (glfwTerminate)
        return 1
    }
    
    # 5. Now you can use OpenGL functions!
    (glClearColor 0.2 0.3 0.4 1.0)
    
    # Main loop
    while (== (glfwWindowShouldClose window) 0) {
        (glClear GL_COLOR_BUFFER_BIT)
        
        # Your OpenGL rendering here...
        
        (glfwSwapBuffers window)
        (glfwPollEvents)
    }
    
    (glfwTerminate)
    return 0
}
```

## Common OpenGL Functions

After `glewInit()`, these functions are available:

### Clearing and Setup
- `glClear(mask)` - Clear buffers
- `glClearColor(r, g, b, a)` - Set clear color
- `glViewport(x, y, width, height)` - Set viewport

### Immediate Mode Drawing (Legacy OpenGL)
- `glBegin(mode)` - Start primitive
- `glEnd()` - Finish primitive
- `glVertex2f(x, y)` - 2D vertex
- `glVertex3f(x, y, z)` - 3D vertex
- `glColor3f(r, g, b)` - Set color

### Matrix Operations
- `glMatrixMode(mode)` - Set active matrix
- `glLoadIdentity()` - Reset to identity
- `glOrtho(left, right, bottom, top, near, far)` - Orthographic projection

## OpenGL Constants

Define these in your code:

```nano
# Clear buffers
let GL_COLOR_BUFFER_BIT: int = 0x00004000
let GL_DEPTH_BUFFER_BIT: int = 0x00000100

# Drawing modes  
let GL_TRIANGLES: int = 0x0004
let GL_QUADS: int = 0x0007
let GL_LINES: int = 0x0001

# Matrix modes
let GL_MODELVIEW: int = 0x1700
let GL_PROJECTION: int = 0x1701
```

## Troubleshooting

**Error: "Failed to initialize GLEW"**
- Make sure you called `glfwMakeContextCurrent()` before `glewInit()`
- Verify OpenGL drivers are installed

**Error: "Symbol not found" when linking**
- Check that GLEW is installed: `brew list glew` or `dpkg -l | grep glew`
- Verify pkg-config can find it: `pkg-config --libs glew`

## Examples

See `examples/` for OpenGL demos using GLFW + GLEW.

