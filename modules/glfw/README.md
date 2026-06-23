# GLFW Module for nanolang

Modern OpenGL window and input library for creating cross-platform windowed applications with OpenGL contexts.

## Installation

### macOS
```bash
brew install glfw
```

### Linux (Ubuntu/Debian)
```bash
sudo apt install libglfw3-dev
```

### Linux (Fedora/RHEL)
```bash
sudo dnf install glfw-devel
```

## Usage

```nano
import "modules/glfw/glfw.nano"

fn main() -> int {
    # Initialize GLFW
    if (== (glfwInit) 0) {
        (println "Failed to initialize GLFW")
        return 1
    }
    
    # Create window
    let window: int = (glfwCreateWindow 800 600 "My OpenGL Window" 0 0)
    if (== window 0) {
        (println "Failed to create window")
        (glfwTerminate)
        return 1
    }
    
    # Make OpenGL context current
    (glfwMakeContextCurrent window)
    
    # Main loop
    while (== (glfwWindowShouldClose window) 0) {
        # Render here...
        
        (glfwSwapBuffers window)
        (glfwPollEvents)
    }
    
    # Cleanup
    (glfwDestroyWindow window)
    (glfwTerminate)
    return 0
}
```

## Features

- **Cross-platform**: Works on macOS, Linux, Windows
- **Modern OpenGL**: Supports OpenGL 3.0+
- **Input handling**: Keyboard, mouse, joystick
- **Window management**: Resizing, fullscreen, multi-monitor
- **High DPI support**: Automatic scaling on retina displays

## API Reference

### Initialization
- `glfwInit()` - Initialize GLFW library
- `glfwTerminate()` - Cleanup and shutdown

### Window Management  
- `glfwCreateWindow(width, height, title, monitor, share)` - Create window
- `glfwDestroyWindow(window)` - Destroy window
- `glfwWindowShouldClose(window)` - Check if window should close
- `glfwSetWindowShouldClose(window, value)` - Set close flag

### Rendering
- `glfwMakeContextCurrent(window)` - Make context current
- `glfwSwapBuffers(window)` - Swap front/back buffers
- `glfwGetFramebufferSize(window, &width, &height)` - Get pixel dimensions

### Input
- `glfwPollEvents()` - Process pending events
- `glfwGetKey(window, key)` - Get key state
- `glfwGetMouseButton(window, button)` - Get mouse button state
- `glfwGetCursorPos(window, &x, &y)` - Get cursor position

### Timing
- `glfwGetTime()` - Get time since init (seconds)
- `glfwSetTime(time)` - Set time counter

## See Also

- [GLFW Official Documentation](https://www.glfw.org/documentation.html)
- [OpenGL Tutorial](https://learnopengl.com/)
- GLEW module (for OpenGL function loading)

