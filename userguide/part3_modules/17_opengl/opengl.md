# 17.1 OpenGL - Drawing Commands

**Issue GPU drawing commands: primitives, colors, transformations, and the matrix stack.**

The `modules/opengl/opengl.nano` module exposes OpenGL drawing functions directly in NanoLang. These are the functions that actually describe geometry to the GPU — vertices, colors, normals, and the transforms that position objects in 3D space.

This module assumes GLFW (for the window and context) and GLEW (for extension loading) have already been initialized. See [Chapter 17 Overview](index.html) for the initialization sequence.

## Immediate Mode Drawing

OpenGL's immediate mode lets you describe geometry inline by calling `glBegin`, submitting vertices, then calling `glEnd`. While deprecated in modern core profiles, it is the simplest way to draw shapes and is fully supported in compatibility contexts.

### Primitive Types

Pass a primitive type constant to `glBegin`:

| Constant | Value | Shape |
|----------|-------|-------|
| `GL_POINTS` | 0 | One point per vertex |
| `GL_LINES` | 1 | One line per pair of vertices |
| `GL_LINE_LOOP` | 2 | Connected lines, closing back to first |
| `GL_LINE_STRIP` | 3 | Connected lines, open |
| `GL_TRIANGLES` | 4 | One triangle per 3 vertices |
| `GL_TRIANGLE_STRIP` | 5 | Shared-edge triangles |
| `GL_TRIANGLE_FAN` | 6 | Fan of triangles around first vertex |
| `GL_QUADS` | 7 | One quad per 4 vertices |
| `GL_POLYGON` | 9 | Convex polygon |

```nano
from "modules/opengl/opengl.nano" import
    glBegin, glEnd, glVertex2f, glVertex3f, glColor3f, glColor4f,
    GL_TRIANGLES, GL_QUADS, GL_LINES, GL_LINE_LOOP

# Triangle
(glBegin GL_TRIANGLES)
    (glColor3f 1.0 0.0 0.0)
    (glVertex2f 0.0 0.5)
    (glColor3f 0.0 1.0 0.0)
    (glVertex2f -0.5 -0.5)
    (glColor3f 0.0 0.0 1.0)
    (glVertex2f 0.5 -0.5)
(glEnd)
```

### Vertices

Use `glVertex2f` for 2D coordinates (z=0) or `glVertex3f` for full 3D:

```nano
(glVertex2f 0.0 0.5)           # 2D
(glVertex3f 0.5 -0.5 0.0)     # 3D
```

Coordinates are in _normalized device coordinates_ (NDC) by default: -1.0 to +1.0 in X and Y, with the center of the screen at (0, 0). Apply a projection matrix to use pixel coordinates or perspective.

### Colors

```nano
from "modules/opengl/opengl.nano" import glColor3f, glColor4f

(glColor3f 1.0 0.5 0.0)         # RGB: orange
(glColor4f 0.0 0.8 1.0 0.5)    # RGBA: semi-transparent cyan
```

Color components are 0.0–1.0. The current color is applied to all subsequent vertices until changed.

## Clearing the Screen

```nano
from "modules/glew/glew.nano" import glClear, nlg_glClearColor, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT

(nlg_glClearColor 0.1 0.1 0.2 1.0)   # set clear color (dark blue)
(glClear GL_COLOR_BUFFER_BIT)          # clear color buffer
(glClear (+ GL_COLOR_BUFFER_BIT GL_DEPTH_BUFFER_BIT))  # also clear depth
```

## The Matrix Stack

OpenGL uses a matrix stack to transform objects. The two key matrices are:

- **Model-View** (`GL_MODELVIEW`): positions and orients the camera and objects in the scene
- **Projection** (`GL_PROJECTION`): maps 3D scene to 2D screen (perspective or orthographic)

```nano
from "modules/glew/glew.nano" import
    glMatrixMode, glLoadIdentity, glOrtho, glFrustum,
    glPushMatrix, glPopMatrix,
    GL_MODELVIEW, GL_PROJECTION

# Set up an orthographic projection (2D-style, no perspective)
(glMatrixMode GL_PROJECTION)
(glLoadIdentity)
(glOrtho -1.0 1.0 -1.0 1.0 -1.0 1.0)   # left right bottom top near far

# Switch to modelview for object transforms
(glMatrixMode GL_MODELVIEW)
(glLoadIdentity)
```

### Transformations

All transforms operate on the current matrix:

```nano
from "modules/opengl/opengl.nano" import glTranslatef, glRotatef, glScalef

(glTranslatef 0.5 0.2 0.0)         # move right 0.5, up 0.2
(glRotatef 45.0 0.0 0.0 1.0)      # rotate 45 degrees around Z axis
(glScalef 2.0 2.0 1.0)            # double size in X and Y
```

`glRotatef(angle, x, y, z)` rotates `angle` degrees around the axis vector `(x, y, z)`. Common axes:
- `(0, 0, 1)` — Z axis (2D rotation, spinning in-plane)
- `(0, 1, 0)` — Y axis (spinning left-right)
- `(1, 0, 0)` — X axis (pitching up-down)

### Push and Pop

Use `glPushMatrix` / `glPopMatrix` to save and restore the matrix state — essential for hierarchical scenes:

```nano
from "modules/glew/glew.nano" import glPushMatrix, glPopMatrix

# Save the current transform
(glPushMatrix)

    # Apply a temporary transform for this object
    (glTranslatef 0.3 0.0 0.0)
    (glRotatef 30.0 0.0 0.0 1.0)

    # Draw object here...
    (glBegin GL_TRIANGLES)
        (glVertex2f 0.0 0.1)
        (glVertex2f -0.1 -0.1)
        (glVertex2f 0.1 -0.1)
    (glEnd)

# Restore the matrix to what it was before
(glPopMatrix)
```

## Drawing Style

### Line Width

```nano
from "modules/opengl/opengl.nano" import glLineWidth

(glLineWidth 3.0)   # 3-pixel lines
```

### Point Size

```nano
from "modules/opengl/opengl.nano" import glPointSize

(glPointSize 5.0)   # 5-pixel points
```

### Polygon Mode (Wireframe)

```nano
from "modules/glew/glew.nano" import glPolygonMode, GL_FRONT_AND_BACK, GL_LINE, GL_FILL

(glPolygonMode GL_FRONT_AND_BACK GL_LINE)   # wireframe
(glPolygonMode GL_FRONT_AND_BACK GL_FILL)   # solid (default)
```

## Enabling OpenGL Capabilities

Many OpenGL features are disabled by default and must be enabled explicitly:

```nano
from "modules/glew/glew.nano" import glEnable, glDisable,
    GL_DEPTH_TEST, GL_BLEND, GL_LIGHTING, GL_CULL_FACE, GL_NORMALIZE

(glEnable GL_DEPTH_TEST)    # Enable Z-buffering for 3D scenes
(glEnable GL_BLEND)         # Enable alpha blending
(glEnable GL_CULL_FACE)     # Skip back-facing triangles (optimization)
(glEnable GL_NORMALIZE)     # Auto-normalize normals (needed if scaled)
```

For blending, set the blend equation:

```nano
from "modules/glew/glew.nano" import glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA

(glBlendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA)   # standard alpha blending
```

## Lighting (Fixed-Function)

OpenGL's fixed-function lighting pipeline:

```nano
from "modules/glew/glew.nano" import
    glEnable, glShadeModel, glColorMaterial, glDepthFunc,
    nl_glLightfv4, nl_glMaterialfv4, nlg_glMaterialf,
    GL_LIGHTING, GL_LIGHT0, GL_COLOR_MATERIAL, GL_SMOOTH,
    GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_SPECULAR,
    GL_FRONT, GL_AMBIENT_AND_DIFFUSE, GL_SHININESS, GL_LESS

from "modules/opengl/opengl.nano" import glNormal3f

# Enable lighting
(glEnable GL_LIGHTING)
(glEnable GL_LIGHT0)
(glEnable GL_COLOR_MATERIAL)
(glShadeModel GL_SMOOTH)

# Position light at (5, 10, 5, 1) — point light (w=1)
(nl_glLightfv4 GL_LIGHT0 GL_POSITION 5.0 10.0 5.0 1.0)

# Set light color
(nl_glLightfv4 GL_LIGHT0 GL_DIFFUSE 1.0 1.0 0.9 1.0)
(nl_glLightfv4 GL_LIGHT0 GL_AMBIENT 0.1 0.1 0.1 1.0)

# Enable depth testing
(glEnable GL_DEPTH_TEST)
(glDepthFunc GL_LESS)

# When drawing geometry, supply normals for correct lighting:
(glNormal3f 0.0 1.0 0.0)   # upward-facing normal
(glVertex3f 0.0 0.0 0.0)
```

## Normals

Normals tell OpenGL which direction a surface faces — required for lighting calculations:

```nano
from "modules/opengl/opengl.nano" import glNormal3f

(glBegin GL_TRIANGLES)
    (glNormal3f 0.0 0.0 1.0)     # face points toward +Z (toward viewer)
    (glVertex3f -0.5 -0.5 0.0)
    (glVertex3f 0.5 -0.5 0.0)
    (glVertex3f 0.0 0.5 0.0)
(glEnd)
```

## Modern Pipeline: Shaders and VBOs

For modern OpenGL (3.x+), use the `nl_gl3_*` helpers from the GLEW module (see [17.3 GLEW](glew.html)). Here is a taste:

```nano
from "modules/glew/glew.nano" import
    nl_gl3_create_program_from_sources,
    nl_gl3_use_program,
    nl_gl3_gen_vertex_array, nl_gl3_bind_vertex_array,
    nl_gl3_gen_buffer, nl_gl3_bind_buffer, nl_gl3_buffer_data_f32,
    nl_gl3_enable_vertex_attrib_array, nl_gl3_vertex_attrib_pointer_f32,
    nl_gl3_draw_arrays,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_TRIANGLES

let vert_src: string = "
    #version 330 core
    layout(location = 0) in vec3 pos;
    void main() { gl_Position = vec4(pos, 1.0); }
"

let frag_src: string = "
    #version 330 core
    out vec4 color;
    void main() { color = vec4(1.0, 0.5, 0.0, 1.0); }
"

let program: int = (nl_gl3_create_program_from_sources vert_src frag_src)

let verts: array<float> = [0.0, 0.5, 0.0,  -0.5, -0.5, 0.0,  0.5, -0.5, 0.0]
let vao: int = (nl_gl3_gen_vertex_array)
let vbo: int = (nl_gl3_gen_buffer)

(nl_gl3_bind_vertex_array vao)
(nl_gl3_bind_buffer GL_ARRAY_BUFFER vbo)
(nl_gl3_buffer_data_f32 GL_ARRAY_BUFFER verts GL_STATIC_DRAW)
(nl_gl3_enable_vertex_attrib_array 0)
(nl_gl3_vertex_attrib_pointer_f32 0 3 0 12 0)   # attr 0, 3 floats, stride 12 bytes

# In render loop:
(nl_gl3_use_program program)
(nl_gl3_bind_vertex_array vao)
(nl_gl3_draw_arrays GL_TRIANGLES 0 3)
```

See [17.3 GLEW](glew.html) for the full modern pipeline API.

## Complete Example: Spinning Colored Triangle

```nano
from "modules/glfw/glfw.nano" import
    glfwInit, glfwTerminate, glfwCreateWindow, glfwDestroyWindow,
    glfwMakeContextCurrent, glfwSwapBuffers, glfwPollEvents,
    glfwWindowShouldClose, glfwSwapInterval, glfwGetKey,
    glfwSetWindowShouldClose, glfwGetTime,
    GLFWwindow, GLFWmonitor, GLFW_PRESS, GLFW_KEY_ESCAPE

from "modules/glew/glew.nano" import
    glewInit, glClear, nlg_glClearColor,
    glMatrixMode, glLoadIdentity, glPushMatrix, glPopMatrix,
    GL_COLOR_BUFFER_BIT, GL_MODELVIEW, GLEW_OK

from "modules/opengl/opengl.nano" import
    glBegin, glEnd, glVertex2f, glColor3f, glRotatef, GL_TRIANGLES

fn main() -> void {
    (glfwInit)
    (glfwSwapInterval 1)

    let window: GLFWwindow = (glfwCreateWindow 640 480 "Spinning Triangle" ...)

    (glfwMakeContextCurrent window)
    (glewInit)

    while (== (glfwWindowShouldClose window) 0) {
        let esc: int = (glfwGetKey window GLFW_KEY_ESCAPE)
        if (== esc GLFW_PRESS) {
            (glfwSetWindowShouldClose window 1)
        }

        (nlg_glClearColor 0.08 0.08 0.15 1.0)
        (glClear GL_COLOR_BUFFER_BIT)

        (glMatrixMode GL_MODELVIEW)
        (glLoadIdentity)

        # Rotate based on elapsed time
        let t: float = (glfwGetTime)
        let angle: float = (* t 90.0)   # 90 degrees per second
        (glRotatef angle 0.0 0.0 1.0)

        (glBegin GL_TRIANGLES)
            (glColor3f 1.0 0.2 0.2)
            (glVertex2f 0.0 0.5)
            (glColor3f 0.2 1.0 0.2)
            (glVertex2f -0.43 -0.25)
            (glColor3f 0.2 0.2 1.0)
            (glVertex2f 0.43 -0.25)
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
| `glBegin(mode)` | Begin a primitive group |
| `glEnd()` | End a primitive group |
| `glVertex2f(x, y)` | Submit a 2D vertex |
| `glVertex3f(x, y, z)` | Submit a 3D vertex |
| `glColor3f(r, g, b)` | Set current color (RGB) |
| `glColor4f(r, g, b, a)` | Set current color (RGBA) |
| `glNormal3f(nx, ny, nz)` | Set current normal vector |
| `glTranslatef(x, y, z)` | Translate current matrix |
| `glRotatef(angle, x, y, z)` | Rotate current matrix |
| `glScalef(x, y, z)` | Scale current matrix |
| `glLineWidth(w)` | Set line drawing width |
| `glPointSize(s)` | Set point drawing size |
| `glMaterialf(face, pname, param)` | Set material property |

(Matrix, state, and modern pipeline functions are in the GLEW module — see [17.3 GLEW](glew.html).)

---

**Previous:** [Chapter 17 Overview](index.html)
**Next:** [17.2 GLFW](glfw.html)
