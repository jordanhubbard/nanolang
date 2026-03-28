# 17.2 GLFW - Window and Input Management

**Create OpenGL windows, manage contexts, and handle keyboard and mouse input.**

GLFW (Graphics Library Framework) is a lightweight library for creating windows with OpenGL contexts and receiving input events. It is the modern replacement for GLUT's windowing functionality. NanoLang exposes GLFW through `modules/glfw/glfw.nano`.

## Overview

GLFW provides:

- Cross-platform window creation with an embedded OpenGL context
- Context version and profile hints (for requesting OpenGL 3.3 core, 4.1, etc.)
- Per-frame event polling (keyboard, mouse buttons, cursor position)
- Swap buffer control (vsync, swap interval)
- High-precision time (`glfwGetTime`)
- Window close detection

## Initialization

```nano
from "modules/glfw/glfw.nano" import glfwInit, glfwTerminate

fn main() -> void {
    let ok: int = (glfwInit)
    if (== ok 0) {
        (print "Failed to initialize GLFW")
        return
    }

    # ... create windows, render, etc. ...

    (glfwTerminate)
}

shadow main { assert true }
```

Always call `glfwTerminate` before exiting to release OS resources.

## Window Hints

Call `glfwWindowHint` before `glfwCreateWindow` to configure the OpenGL context. Hints remain set until cleared or GLFW is re-initialized.

```nano
from "modules/glfw/glfw.nano" import glfwWindowHint

# Request OpenGL 3.3 core profile
(glfwWindowHint 0x00020001 3)   # GLFW_CONTEXT_VERSION_MAJOR
(glfwWindowHint 0x00020002 3)   # GLFW_CONTEXT_VERSION_MINOR
(glfwWindowHint 0x00022001 1)   # GLFW_OPENGL_CORE_PROFILE (value 1)
(glfwWindowHint 0x00022008 1)   # GLFW_OPENGL_FORWARD_COMPAT (for macOS)
```

For legacy/immediate-mode programs, request a compatibility context:

```nano
# OpenGL 2.1 compatibility (immediate mode supported)
(glfwWindowHint 0x00020001 2)
(glfwWindowHint 0x00020002 1)
```

> **Note:** GLFW hint integer constants are not yet exported as named NanoLang constants — use their raw numeric values (as shown above), or define them as local `let` bindings.

## Creating a Window

`glfwCreateWindow` opens a window with an embedded OpenGL context. For windowed mode, pass `null` (opaque zero-value) for the monitor and share parameters.

```nano
from "modules/glfw/glfw.nano" import glfwCreateWindow, glfwDestroyWindow,
    GLFWwindow, GLFWmonitor

fn open_window(title: string, width: int, height: int) -> GLFWwindow {
    # GLFWmonitor and GLFWwindow null values must be constructed via C shim
    # or runtime helpers; here shown conceptually
    let no_monitor: GLFWmonitor = (make_null_glfw_monitor)
    let no_share: GLFWwindow = (make_null_glfw_window)
    let window: GLFWwindow = (glfwCreateWindow width height title no_monitor no_share)
    return window
}

shadow open_window { assert true }
```

Destroy the window when done:

```nano
(glfwDestroyWindow window)
```

## Making the Context Current

Before calling any OpenGL or GLEW functions, make the window's context current on the calling thread:

```nano
from "modules/glfw/glfw.nano" import glfwMakeContextCurrent

(glfwMakeContextCurrent window)
```

Only one context can be current per thread. If you have multiple windows, switch contexts when rendering to each.

## VSync and Swap Interval

```nano
from "modules/glfw/glfw.nano" import glfwSwapInterval

(glfwSwapInterval 1)   # enable vsync (swap every monitor refresh)
(glfwSwapInterval 0)   # disable vsync (run as fast as possible)
(glfwSwapInterval 2)   # swap every 2 refreshes (30 fps on a 60 Hz display)
```

Call this after making the context current.

## Swapping Buffers

GLFW uses double buffering. After drawing a frame with OpenGL, call `glfwSwapBuffers` to display it:

```nano
from "modules/glfw/glfw.nano" import glfwSwapBuffers

(glfwSwapBuffers window)
```

## Event Polling

`glfwPollEvents` processes all pending window and input events. Call it once per frame in the main loop:

```nano
from "modules/glfw/glfw.nano" import glfwPollEvents

(glfwPollEvents)
```

## Window Close Detection

```nano
from "modules/glfw/glfw.nano" import glfwWindowShouldClose, glfwSetWindowShouldClose

let mut running: bool = true
while running {
    if (!= (glfwWindowShouldClose window) 0) {
        set running false
    }
    # ... render ...
    (glfwSwapBuffers window)
    (glfwPollEvents)
}

# Or: signal close from within the loop (e.g., on ESC key)
(glfwSetWindowShouldClose window 1)
```

## Keyboard Input

`glfwGetKey` returns the current state of a key: `GLFW_PRESS` (1), `GLFW_RELEASE` (0), or `GLFW_REPEAT` (2).

```nano
from "modules/glfw/glfw.nano" import glfwGetKey,
    GLFW_PRESS, GLFW_RELEASE, GLFW_REPEAT,
    GLFW_KEY_ESCAPE, GLFW_KEY_SPACE,
    GLFW_KEY_LEFT, GLFW_KEY_RIGHT, GLFW_KEY_UP, GLFW_KEY_DOWN,
    GLFW_KEY_R

fn handle_input(window: GLFWwindow) -> void {
    # Close on ESC
    if (== (glfwGetKey window GLFW_KEY_ESCAPE) GLFW_PRESS) {
        (glfwSetWindowShouldClose window 1)
    }

    # Movement
    if (== (glfwGetKey window GLFW_KEY_LEFT) GLFW_PRESS) {
        (print "moving left")
    }
    if (== (glfwGetKey window GLFW_KEY_RIGHT) GLFW_PRESS) {
        (print "moving right")
    }

    # Reset
    if (== (glfwGetKey window GLFW_KEY_R) GLFW_PRESS) {
        (print "reset")
    }
}

shadow handle_input { assert true }
```

**Keyboard constants:**

| Constant | Key |
|----------|-----|
| `GLFW_KEY_ESCAPE` | Escape |
| `GLFW_KEY_SPACE` | Spacebar |
| `GLFW_KEY_LEFT` | Left arrow |
| `GLFW_KEY_RIGHT` | Right arrow |
| `GLFW_KEY_UP` | Up arrow |
| `GLFW_KEY_DOWN` | Down arrow |
| `GLFW_KEY_R` | R key |
| `GLFW_KEY_1` .. `GLFW_KEY_6` | Number row 1–6 |
| `GLFW_KEY_MINUS` | Minus/dash |
| `GLFW_KEY_EQUAL` | Equals |
| `GLFW_KEY_KP_SUBTRACT` | Numpad minus |
| `GLFW_KEY_KP_ADD` | Numpad plus |

## Mouse Input

### Mouse Buttons

```nano
from "modules/glfw/glfw.nano" import glfwGetMouseButton,
    GLFW_MOUSE_BUTTON_LEFT, GLFW_MOUSE_BUTTON_RIGHT, GLFW_MOUSE_BUTTON_MIDDLE,
    GLFW_PRESS

let lmb: int = (glfwGetMouseButton window GLFW_MOUSE_BUTTON_LEFT)
if (== lmb GLFW_PRESS) {
    (print "left button held")
}
```

### Cursor Position

`glfwGetCursorPos` writes X and Y into output pointers. Use a C shim to read the actual values:

```nano
from "modules/glfw/glfw.nano" import glfwGetCursorPos

# With C shim providing output pointers:
(glfwGetCursorPos window xpos_ptr ypos_ptr)
```

## Timing

`glfwGetTime` returns seconds elapsed since GLFW was initialized, as a `float`:

```nano
from "modules/glfw/glfw.nano" import glfwGetTime, glfwSetTime

let t: float = (glfwGetTime)

# Reset the timer
(glfwSetTime 0.0)
```

Use the timer for animation, delta-time calculations, and FPS measurement:

```nano
let mut last_time: float = (glfwGetTime)

# In the render loop:
let now: float = (glfwGetTime)
let dt: float = (- now last_time)
set last_time now

# dt is elapsed seconds since last frame (~0.016 at 60 fps)
```

## Framebuffer Size

On high-DPI (Retina) displays, the framebuffer may be larger than the window in screen coordinates. Always use the framebuffer size for `glViewport`:

```nano
from "modules/glfw/glfw.nano" import glfwGetFramebufferSize
from "modules/glew/glew.nano" import glViewport

# With C shim providing output pointers for width and height:
(glfwGetFramebufferSize window width_ptr height_ptr)
(glViewport 0 0 fb_width fb_height)
```

## Complete Game Loop Example

```nano
from "modules/glfw/glfw.nano" import
    glfwInit, glfwTerminate,
    glfwWindowHint, glfwCreateWindow, glfwDestroyWindow,
    glfwMakeContextCurrent, glfwSwapBuffers, glfwPollEvents,
    glfwWindowShouldClose, glfwSwapInterval,
    glfwGetKey, glfwSetWindowShouldClose, glfwGetTime,
    GLFWwindow, GLFWmonitor,
    GLFW_PRESS, GLFW_KEY_ESCAPE, GLFW_KEY_LEFT, GLFW_KEY_RIGHT

from "modules/glew/glew.nano" import
    glewInit, glClear, nlg_glClearColor, GL_COLOR_BUFFER_BIT, GLEW_OK

from "modules/opengl/opengl.nano" import
    glBegin, glEnd, glVertex2f, glColor3f, glTranslatef,
    glMatrixMode, GL_TRIANGLES, GL_MODELVIEW

fn main() -> void {
    (glfwInit)
    (glfwWindowHint 0x00020001 2)   # OpenGL 2.1
    (glfwWindowHint 0x00020002 1)

    let window: GLFWwindow = (glfwCreateWindow 800 600 "GLFW Demo" ...)
    (glfwMakeContextCurrent window)
    (glfwSwapInterval 1)

    let glew_err: int = (glewInit)
    if (!= glew_err GLEW_OK) {
        (print "GLEW failed")
        (glfwTerminate)
        return
    }

    let mut player_x: float = 0.0
    let mut last_t: float = (glfwGetTime)

    while (== (glfwWindowShouldClose window) 0) {
        let now: float = (glfwGetTime)
        let dt: float = (- now last_t)
        set last_t now

        # Input
        if (== (glfwGetKey window GLFW_KEY_ESCAPE) GLFW_PRESS) {
            (glfwSetWindowShouldClose window 1)
        }
        if (== (glfwGetKey window GLFW_KEY_LEFT) GLFW_PRESS) {
            set player_x (- player_x (* 1.5 dt))
        }
        if (== (glfwGetKey window GLFW_KEY_RIGHT) GLFW_PRESS) {
            set player_x (+ player_x (* 1.5 dt))
        }

        # Render
        (nlg_glClearColor 0.05 0.05 0.1 1.0)
        (glClear GL_COLOR_BUFFER_BIT)

        (glMatrixMode GL_MODELVIEW)
        (glTranslatef player_x 0.0 0.0)

        (glBegin GL_TRIANGLES)
            (glColor3f 0.0 1.0 0.5)
            (glVertex2f 0.0 0.15)
            (glVertex2f -0.1 -0.1)
            (glVertex2f 0.1 -0.1)
        (glEnd)

        (glfwSwapBuffers window)
        (glfwPollEvents)
    }

    (glfwDestroyWindow window)
    (glfwTerminate)
}

shadow main { assert true }
```

## API Summary

| Function | Description |
|----------|-------------|
| `glfwInit()` | Initialize GLFW |
| `glfwTerminate()` | Shut down GLFW |
| `glfwWindowHint(hint, value)` | Set context creation hint |
| `glfwCreateWindow(w, h, title, monitor, share)` | Create window + GL context |
| `glfwDestroyWindow(window)` | Close and destroy window |
| `glfwWindowShouldClose(window)` | 1 if close was requested |
| `glfwSetWindowShouldClose(window, value)` | Signal close from code |
| `glfwMakeContextCurrent(window)` | Activate GL context on thread |
| `glfwSwapBuffers(window)` | Present rendered frame |
| `glfwPollEvents()` | Process pending events |
| `glfwSwapInterval(interval)` | Set vsync interval |
| `glfwGetFramebufferSize(window, w_out, h_out)` | Get framebuffer dimensions |
| `glfwGetKey(window, key)` | Query key state |
| `glfwGetMouseButton(window, button)` | Query mouse button state |
| `glfwGetCursorPos(window, x_out, y_out)` | Get cursor position |
| `glfwGetTime()` | Seconds since glfwInit |
| `glfwSetTime(time)` | Reset timer |

---

**Previous:** [17.1 OpenGL](opengl.html)
**Next:** [17.3 GLEW](glew.html)
