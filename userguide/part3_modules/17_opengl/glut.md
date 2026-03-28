# 17.4 GLUT - OpenGL Utility Toolkit

**Built-in 3D shapes, bitmap text, and a simple event-driven main loop.**

GLUT (OpenGL Utility Toolkit) — and its open-source successor FreeGLUT — provides a convenient set of utilities for writing simple OpenGL programs. Its primary features are built-in geometric shapes (teapot, sphere, cube, cone, torus, and more), bitmap font rendering, and a callback-based main loop.

NanoLang exposes GLUT through `modules/glut/glut.nano`.

> **GLUT vs. GLFW:** GLUT's main loop (`glutMainLoop`) is older and less flexible than GLFW's per-frame loop. For new projects, prefer GLFW (see [17.2 GLFW](glfw.html)). Use GLUT when you want quick access to its built-in shapes, or when working with legacy code.

## Overview

GLUT provides:

- Window setup: `glutCreateWindow`, `glutInitWindowSize`, `glutInitDisplayMode`
- Event loop: `glutMainLoop` with callbacks for display, reshape, keyboard
- Built-in solid and wireframe shapes: teapot, sphere, cube, cone, torus, dodecahedron, octahedron, tetrahedron, icosahedron
- Bitmap text rendering: `glutBitmapCharacter`, `glutBitmapLength`
- State queries: `glutGet`

## Initialization

```nano
from "modules/glut/glut.nano" import
    glutInit, glutInitDisplayMode, glutInitWindowSize, glutInitWindowPosition,
    glutCreateWindow,
    GLUT_DOUBLE, GLUT_RGBA, GLUT_DEPTH

fn setup_glut() -> int {
    # argc/argv pointers — pass 0 for no args
    (glutInit 0 0)

    # Double-buffered RGBA window with depth buffer
    let mode: int = (+ GLUT_DOUBLE (+ GLUT_RGBA GLUT_DEPTH))
    (glutInitDisplayMode mode)

    (glutInitWindowSize 800 600)
    (glutInitWindowPosition 100 100)

    let window_id: int = (glutCreateWindow "GLUT Demo")
    return window_id
}

shadow setup_glut { assert true }
```

**Display mode flags:**

| Constant | Value | Description |
|----------|-------|-------------|
| `GLUT_RGB` / `GLUT_RGBA` | 0 | RGB(A) color mode |
| `GLUT_SINGLE` | 0 | Single buffer |
| `GLUT_DOUBLE` | 2 | Double buffer (use for animation) |
| `GLUT_DEPTH` | 16 | Depth buffer (for 3D) |
| `GLUT_ALPHA` | 8 | Alpha channel |
| `GLUT_STENCIL` | 32 | Stencil buffer |
| `GLUT_ACCUM` | 4 | Accumulation buffer |

## The GLUT Main Loop

In the GLUT model, you register callback functions for events, then call `glutMainLoop()` which never returns. This is different from GLFW's explicit-loop style.

```nano
from "modules/glut/glut.nano" import glutMainLoop, glutSwapBuffers, glutPostRedisplay

# Register callbacks (requires C shims for function pointers in NanoLang)
# See note below about callbacks.

# Start the event loop (blocks forever)
(glutMainLoop)
```

> **Note on callbacks:** `glutDisplayFunc`, `glutReshapeFunc`, and `glutKeyboardFunc` accept C function pointers, which NanoLang cannot pass directly. In practice, define the callback bodies in a C shim file and register them from NanoLang via a helper function, or use GLFW instead.

### Manual Redisplay

From within a callback (or a C shim), trigger a redraw:

```nano
from "modules/glut/glut.nano" import glutPostRedisplay

# Request that the display callback be called again
(glutPostRedisplay)
```

### Buffer Swap

After drawing in the display callback:

```nano
from "modules/glut/glut.nano" import glutSwapBuffers

(glutSwapBuffers)
```

## Built-in Solid Shapes

GLUT provides pre-built geometry for common shapes. Solid variants include normals (for lighting); wireframe variants draw edges only.

### Teapot (the classic OpenGL demo)

```nano
from "modules/glut/glut.nano" import glutSolidTeapot, glutWireTeapot

(glutSolidTeapot 1.0)    # solid teapot with radius 1.0
(glutWireTeapot 1.0)     # wireframe teapot
```

The teapot is the "Hello, World" of 3D graphics. Combine it with a rotating transform:

```nano
from "modules/opengl/opengl.nano" import glRotatef
from "modules/glut/glut.nano" import glutSolidTeapot

(glRotatef 45.0 0.0 1.0 0.0)   # rotate 45 degrees around Y
(glutSolidTeapot 0.8)
```

### Sphere

```nano
from "modules/glut/glut.nano" import glutSolidSphere, glutWireSphere

# radius=0.5, 32 longitude slices, 32 latitude stacks
(glutSolidSphere 0.5 32 32)
(glutWireSphere 0.5 16 16)
```

Higher slice/stack counts produce smoother spheres at the cost of more geometry.

### Cube

```nano
from "modules/glut/glut.nano" import glutSolidCube, glutWireCube

(glutSolidCube 1.0)    # unit cube (side length 1.0)
(glutWireCube 1.0)
```

### Cone

```nano
from "modules/glut/glut.nano" import glutSolidCone, glutWireCone

# base radius=0.5, height=1.0, 20 slices, 5 stacks
(glutSolidCone 0.5 1.0 20 5)
```

### Torus (Donut)

```nano
from "modules/glut/glut.nano" import glutSolidTorus, glutWireTorus

# innerRadius=0.2, outerRadius=0.5, 16 sides, 32 rings
(glutSolidTorus 0.2 0.5 16 32)
```

### Platonic Solids

```nano
from "modules/glut/glut.nano" import
    glutSolidDodecahedron, glutWireDodecahedron,
    glutSolidOctahedron, glutWireOctahedron,
    glutSolidTetrahedron, glutWireTetrahedron,
    glutSolidIcosahedron, glutWireIcosahedron

(glutSolidDodecahedron)    # 12 pentagonal faces
(glutSolidOctahedron)      # 8 triangular faces
(glutSolidTetrahedron)     # 4 triangular faces
(glutSolidIcosahedron)     # 20 triangular faces
```

## Bitmap Text Rendering

GLUT provides raster fonts for overlaying text directly on the framebuffer. Set a 2D raster position first, then draw characters one at a time.

```nano
from "modules/glut/glut.nano" import
    glutBitmapCharacter, glutBitmapWidth, glutBitmapLength,
    GLUT_BITMAP_HELVETICA_18, GLUT_BITMAP_9_BY_15,
    GLUT_BITMAP_TIMES_ROMAN_24

from "modules/opengl/opengl.nano" import glRasterPos2f

# Position text in 2D (requires orthographic projection)
(glRasterPos2f -0.9 0.9)

# Draw "Hi!" character by character (use a loop in practice)
(glutBitmapCharacter GLUT_BITMAP_HELVETICA_18 72)   # 'H' = ASCII 72
(glutBitmapCharacter GLUT_BITMAP_HELVETICA_18 105)  # 'i' = ASCII 105
(glutBitmapCharacter GLUT_BITMAP_HELVETICA_18 33)   # '!' = ASCII 33
```

**Font constants:**

| Constant | Value | Description |
|----------|-------|-------------|
| `GLUT_BITMAP_9_BY_15` | 2 | Fixed 9×15 pixel font |
| `GLUT_BITMAP_8_BY_13` | 3 | Fixed 8×13 pixel font |
| `GLUT_BITMAP_TIMES_ROMAN_10` | 4 | Times Roman 10pt |
| `GLUT_BITMAP_TIMES_ROMAN_24` | 5 | Times Roman 24pt |
| `GLUT_BITMAP_HELVETICA_10` | 6 | Helvetica 10pt |
| `GLUT_BITMAP_HELVETICA_12` | 7 | Helvetica 12pt |
| `GLUT_BITMAP_HELVETICA_18` | 8 | Helvetica 18pt |

Measure text width before drawing:

```nano
# Width of a single character
let char_w: int = (glutBitmapWidth GLUT_BITMAP_HELVETICA_18 65)   # 'A'

# Total width of a string
let str_w: int = (glutBitmapLength GLUT_BITMAP_HELVETICA_18 "Hello")
```

## Querying Window State

```nano
from "modules/glut/glut.nano" import glutGet

# Window size
let w: int = (glutGet 102)   # GLUT_WINDOW_WIDTH
let h: int = (glutGet 103)   # GLUT_WINDOW_HEIGHT

# Screen size
let sw: int = (glutGet 200)  # GLUT_SCREEN_WIDTH
let sh: int = (glutGet 201)  # GLUT_SCREEN_HEIGHT
```

## Complete Example: Lit Teapot Scene

This example shows what the GLUT callback bodies would contain, assuming they are registered via a C shim:

```nano
from "modules/glew/glew.nano" import
    glewInit, glClear, nlg_glClearColor, glEnable, glShadeModel,
    glMatrixMode, glLoadIdentity, glFrustum, glViewport,
    nl_glLightfv4, nl_glMaterialfv4,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_LIGHTING, GL_LIGHT0, GL_SMOOTH,
    GL_MODELVIEW, GL_PROJECTION,
    GL_POSITION, GL_DIFFUSE, GL_AMBIENT

from "modules/opengl/opengl.nano" import glRotatef, glTranslatef, glColor3f

from "modules/glut/glut.nano" import
    glutInit, glutInitDisplayMode, glutInitWindowSize,
    glutCreateWindow, glutMainLoop, glutSwapBuffers, glutPostRedisplay,
    glutSolidTeapot, glutSolidSphere,
    GLUT_DOUBLE, GLUT_RGBA, GLUT_DEPTH

fn setup() -> void {
    (glutInit 0 0)
    (glutInitDisplayMode (+ GLUT_DOUBLE (+ GLUT_RGBA GLUT_DEPTH)))
    (glutInitWindowSize 800 600)
    (glutCreateWindow "Teapot Scene")
    (glewInit)

    # OpenGL state
    (glEnable GL_DEPTH_TEST)
    (glEnable GL_LIGHTING)
    (glEnable GL_LIGHT0)
    (glShadeModel GL_SMOOTH)

    (nl_glLightfv4 GL_LIGHT0 GL_POSITION 5.0 10.0 5.0 1.0)
    (nl_glLightfv4 GL_LIGHT0 GL_DIFFUSE 1.0 0.9 0.8 1.0)
    (nl_glLightfv4 GL_LIGHT0 GL_AMBIENT 0.1 0.1 0.15 1.0)
}

# This function would be called from the GLUT display callback (via C shim)
fn display(angle: float) -> void {
    (nlg_glClearColor 0.05 0.05 0.1 1.0)
    (glClear (+ GL_COLOR_BUFFER_BIT GL_DEPTH_BUFFER_BIT))

    (glMatrixMode GL_PROJECTION)
    (glLoadIdentity)
    (glFrustum -1.0 1.0 -0.75 0.75 1.0 50.0)

    (glMatrixMode GL_MODELVIEW)
    (glLoadIdentity)
    (glTranslatef 0.0 0.0 -3.5)
    (glRotatef angle 0.0 1.0 0.0)

    # Teapot
    (glColor3f 0.8 0.5 0.2)
    (glutSolidTeapot 0.8)

    # Small sphere off to the side
    (glTranslatef 1.5 0.3 0.0)
    (glColor3f 0.3 0.6 1.0)
    (glutSolidSphere 0.25 24 24)

    (glutSwapBuffers)
    (glutPostRedisplay)
}

shadow setup { assert true }
shadow display { assert true }
```

## Mixing GLUT Shapes with GLFW

Because GLUT shapes are just OpenGL draw calls, you can use them in a GLFW-based application without starting the GLUT event loop:

```nano
# Initialize GLUT just for shape access (no glutMainLoop needed)
(glutInit 0 0)

# Then in your GLFW render loop, call GLUT shapes freely:
(glutSolidTeapot 1.0)
(glutSolidSphere 0.5 32 32)
(glutSolidCube 0.8)
```

This is the most practical way to use GLUT in NanoLang — take the shapes, skip the event loop.

## API Summary

| Function | Description |
|----------|-------------|
| `glutInit(argcp, argv)` | Initialize GLUT |
| `glutInitDisplayMode(mode)` | Set display mode flags |
| `glutInitWindowSize(w, h)` | Set initial window size |
| `glutInitWindowPosition(x, y)` | Set initial window position |
| `glutCreateWindow(title)` | Create window, return ID |
| `glutMainLoop()` | Start GLUT event loop (never returns) |
| `glutSwapBuffers()` | Present rendered frame |
| `glutPostRedisplay()` | Request redraw |
| `glutGet(state)` | Query GLUT/window state |
| `glutSolidTeapot(size)` | Draw solid Utah teapot |
| `glutWireTeapot(size)` | Draw wireframe teapot |
| `glutSolidSphere(radius, slices, stacks)` | Draw solid sphere |
| `glutWireSphere(radius, slices, stacks)` | Draw wireframe sphere |
| `glutSolidCube(size)` | Draw solid cube |
| `glutWireCube(size)` | Draw wireframe cube |
| `glutSolidCone(base, height, slices, stacks)` | Draw solid cone |
| `glutWireCone(base, height, slices, stacks)` | Draw wireframe cone |
| `glutSolidTorus(inner, outer, sides, rings)` | Draw solid torus |
| `glutWireTorus(inner, outer, sides, rings)` | Draw wireframe torus |
| `glutSolidDodecahedron()` | Draw dodecahedron |
| `glutSolidOctahedron()` | Draw octahedron |
| `glutSolidTetrahedron()` | Draw tetrahedron |
| `glutSolidIcosahedron()` | Draw icosahedron |
| `glutBitmapCharacter(font, character)` | Draw one character |
| `glutBitmapWidth(font, character)` | Width of one character |
| `glutBitmapLength(font, str)` | Pixel width of a string |

---

**Previous:** [17.3 GLEW](glew.html)
**Next:** [Chapter 18: Game Development](../18_game_dev/index.html)
