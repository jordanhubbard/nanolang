# Chapter 17: OpenGL Graphics

**Hardware-accelerated 3D graphics with modern OpenGL.**

This chapter covers NanoLang's four OpenGL-related modules and how they work together to produce 3D graphics. If you are coming from SDL (Chapter 16), you already know how to open windows and handle events — here you will learn to hand rendering control over to the OpenGL pipeline and draw geometry directly on the GPU.

## The Four Modules and Their Roles

OpenGL programming involves several cooperating libraries. Understanding the role of each makes the overall system much clearer.

| Module | Library | Role |
|--------|---------|------|
| [17.2 GLFW](glfw.html) | `libglfw` | Create windows, manage OpenGL contexts, poll keyboard/mouse input |
| [17.3 GLEW](glew.html) | `libglew` | Discover and load OpenGL extension functions at runtime |
| [17.1 OpenGL](opengl.html) | `libGL` / system | Issue drawing commands to the GPU |
| [17.4 GLUT](glut.html) | `libglut` / `freeglut` | Legacy utility: built-in shapes, bitmap fonts, simple event loop |

**Typical initialization order:**

1. Call `glfwInit()` (GLFW)
2. Set OpenGL version hints with `glfwWindowHint()` (GLFW)
3. Create a window and context with `glfwCreateWindow()` (GLFW)
4. Make the context current with `glfwMakeContextCurrent()` (GLFW)
5. Call `glewInit()` (GLEW) — this loads all available OpenGL extensions
6. Issue OpenGL drawing commands (`glClear`, `glBegin`, `glVertex3f`, etc.)
7. Swap the back buffer to screen with `glfwSwapBuffers()` (GLFW)

GLUT is an alternative to GLFW for simple programs — it handles steps 1–4 and 7 with a callback-based API. For new code, prefer GLFW.

## Why GLEW Is Needed

OpenGL's core functions are linked at compile time, but most of the interesting modern features (shaders, vertex buffer objects, framebuffers, etc.) are provided as _extensions_ whose function pointers must be loaded at runtime. GLEW does this automatically: after `glewInit()`, all supported extension functions are available to call. Without GLEW, you would need to call `glXGetProcAddress` / `wglGetProcAddress` manually for every modern GL function.

## Immediate Mode vs. Modern OpenGL

NanoLang's OpenGL module supports both styles:

- **Immediate mode** (OpenGL 1.x / 2.x): Call `glBegin(mode)`, then `glVertex3f(x, y, z)` for each vertex, then `glEnd()`. Simple to write, but deprecated in core profiles.
- **Modern (Core Profile)** (OpenGL 3.x+): Upload vertex data to GPU buffers (VBOs), write vertex/fragment shaders in GLSL, then draw with `nl_gl3_draw_arrays`. More complex, but faster and required for compute shaders and modern effects.

For learning and prototyping, immediate mode is fine. The `nl_gl3_*` family of helper functions in the GLEW module makes the VBO/shader path accessible from NanoLang.

## Minimal Window + Triangle Example

This example uses GLFW for windowing, GLEW for extension loading, and OpenGL immediate mode for drawing.

```nano
from "modules/glfw/glfw.nano" import
    glfwInit, glfwTerminate,
    glfwWindowHint, glfwCreateWindow, glfwDestroyWindow,
    glfwMakeContextCurrent, glfwSwapBuffers, glfwPollEvents,
    glfwWindowShouldClose, glfwSwapInterval, glfwGetKey,
    glfwSetWindowShouldClose,
    GLFWwindow, GLFWmonitor,
    GLFW_PRESS, GLFW_KEY_ESCAPE

from "modules/glew/glew.nano" import
    glewInit, glClear, nlg_glClearColor,
    glBegin, glEnd, nlg_glVertex3f, nlg_glColor3f,
    GL_COLOR_BUFFER_BIT, GL_TRIANGLES, GLEW_OK

fn make_null_monitor() -> GLFWmonitor { ... }  # provided by runtime
fn make_null_window() -> GLFWwindow { ... }    # provided by runtime

fn main() -> void {
    let ok: int = (glfwInit)
    if (== ok 0) {
        (print "GLFW init failed")
        return
    }

    # Request OpenGL 2.1 compatibility (works with immediate mode)
    (glfwWindowHint 0x00020001 2)   # GLFW_CONTEXT_VERSION_MAJOR = 2
    (glfwWindowHint 0x00020002 1)   # GLFW_CONTEXT_VERSION_MINOR = 1

    let monitor: GLFWmonitor = (make_null_monitor)
    let share: GLFWwindow = (make_null_window)
    let window: GLFWwindow = (glfwCreateWindow 800 600 "Hello OpenGL" monitor share)

    (glfwMakeContextCurrent window)
    (glfwSwapInterval 1)   # vsync

    let glew_result: int = (glewInit)
    if (!= glew_result GLEW_OK) {
        (print "GLEW init failed")
        (glfwTerminate)
        return
    }

    # Main render loop
    while (== (glfwWindowShouldClose window) 0) {
        # Handle ESC key
        let esc: int = (glfwGetKey window GLFW_KEY_ESCAPE)
        if (== esc GLFW_PRESS) {
            (glfwSetWindowShouldClose window 1)
        }

        # Clear to dark blue
        (nlg_glClearColor 0.1 0.1 0.3 1.0)
        (glClear GL_COLOR_BUFFER_BIT)

        # Draw a colored triangle in immediate mode
        (glBegin GL_TRIANGLES)
            (nlg_glColor3f 1.0 0.0 0.0)   # red
            (nlg_glVertex3f 0.0 0.5 0.0)  # top

            (nlg_glColor3f 0.0 1.0 0.0)   # green
            (nlg_glVertex3f -0.5 -0.5 0.0) # bottom-left

            (nlg_glColor3f 0.0 0.0 1.0)   # blue
            (nlg_glVertex3f 0.5 -0.5 0.0)  # bottom-right
        (glEnd)

        (glfwSwapBuffers window)
        (glfwPollEvents)
    }

    (glfwDestroyWindow window)
    (glfwTerminate)
}

shadow main { assert true }
```

## Sections in This Chapter

- **[17.1 OpenGL](opengl.html)** — Drawing commands, primitives, transformations, the matrix stack, lighting, and the modern shader/VBO pipeline.
- **[17.2 GLFW](glfw.html)** — Window creation, OpenGL context hints, keyboard and mouse input, swap buffers, timing.
- **[17.3 GLEW](glew.html)** — GLEW initialization, extension queries, and the `nl_gl3_*` helpers for shaders, VBOs, textures, and framebuffers.
- **[17.4 GLUT](glut.html)** — The legacy GLUT toolkit: built-in geometric shapes (teapot, sphere, cube), bitmap text, and the `glutMainLoop` event model.

---

**Previous:** [Chapter 16: Graphics Fundamentals](../16_graphics_fundamentals/index.html)
**Next:** [17.1 OpenGL](opengl.html)
