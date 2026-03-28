# 17.3 GLEW - OpenGL Extension Wrangler

**Load OpenGL extensions, manage GPU state, and access the modern OpenGL pipeline.**

GLEW (OpenGL Extension Wrangler Library) solves a fundamental problem: modern OpenGL features (shaders, VBOs, framebuffers, instancing) are exposed as _extensions_ whose function pointers must be discovered at runtime. GLEW handles this automatically — one call to `glewInit()` after context creation makes all supported extensions available.

NanoLang's `modules/glew/glew.nano` exposes GLEW alongside a rich set of core OpenGL functions and the `nl_gl3_*` helpers that make the modern shader/VBO pipeline accessible from NanoLang.

## Overview

The GLEW module provides:

- `glewInit()` — loads all available OpenGL extensions
- `glewIsSupported()` — query extension availability
- Core OpenGL state functions: `glClear`, `glEnable`, `glViewport`, `glBlendFunc`, etc.
- The matrix stack: `glMatrixMode`, `glLoadIdentity`, `glOrtho`, `glPushMatrix`, etc.
- Lighting: `nl_glLightfv4`, `nl_glMaterialfv4`
- Modern pipeline helpers: `nl_gl3_*` for shaders, VBOs, textures, framebuffers

## Why GLEW Is Needed

OpenGL ships with a minimal set of functions linked at build time (OpenGL 1.1 on Windows, OpenGL 2.1 on macOS). Everything beyond that — GLSL shaders, vertex buffer objects, uniform variables, framebuffer objects — lives in extensions. On each platform, the driver exposes these through function pointers that must be loaded with `glXGetProcAddress` (Linux), `wglGetProcAddress` (Windows), or `NSGLGetProcAddress` (macOS).

GLEW hides this complexity. After `glewInit()`, you can call any extension function the driver supports as if it were a regular function.

## Initialization

Call `glewInit()` **after** making an OpenGL context current (via GLFW or SDL):

```nano
from "modules/glfw/glfw.nano" import glfwMakeContextCurrent
from "modules/glew/glew.nano" import glewInit, glewGetErrorString, GLEW_OK

(glfwMakeContextCurrent window)

let err: int = (glewInit)
if (!= err GLEW_OK) {
    let msg: string = (glewGetErrorString err)
    (print msg)
    return
}
```

`GLEW_OK` is 0 — a non-zero return indicates an error.

## Checking Extension Support

Before using a feature, verify the driver supports it:

```nano
from "modules/glew/glew.nano" import glewIsSupported

let has_vbo: int = (glewIsSupported "GL_ARB_vertex_buffer_object")
let has_fbo: int = (glewIsSupported "GL_ARB_framebuffer_object")
let has_instancing: int = (glewIsSupported "GL_ARB_instanced_arrays")

if (== has_vbo 0) {
    (print "VBOs not supported — need a newer driver")
}
```

## Querying Renderer Information

```nano
from "modules/glew/glew.nano" import glGetString, glewGetString

# OpenGL version string (e.g. "4.6.0 NVIDIA 535.183.01")
let version: string = (glGetString 0x1F02)    # GL_VERSION

# Renderer name (e.g. "NVIDIA GeForce RTX 3080")
let renderer: string = (glGetString 0x1F01)   # GL_RENDERER

# GLEW version string
let glew_ver: string = (glewGetString 0x0001) # GLEW_VERSION
```

## Error Checking

```nano
from "modules/glew/glew.nano" import glGetError, GL_NO_ERROR

let e: int = (glGetError)
if (!= e GL_NO_ERROR) {
    (print "OpenGL error detected")
}
```

Call `glGetError` after suspicious operations during development. In production, remove error checks from the inner render loop for performance.

## Core OpenGL State Functions

These core functions are part of the GLEW module in NanoLang:

### Clear

```nano
from "modules/glew/glew.nano" import glClear, nlg_glClearColor,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT

(nlg_glClearColor 0.1 0.1 0.2 1.0)
(glClear GL_COLOR_BUFFER_BIT)
(glClear (+ GL_COLOR_BUFFER_BIT GL_DEPTH_BUFFER_BIT))
```

### Viewport

```nano
from "modules/glew/glew.nano" import glViewport

(glViewport 0 0 800 600)   # x, y, width, height
```

### Enable / Disable

```nano
from "modules/glew/glew.nano" import glEnable, glDisable,
    GL_DEPTH_TEST, GL_BLEND, GL_CULL_FACE, GL_LIGHTING, GL_NORMALIZE

(glEnable GL_DEPTH_TEST)
(glEnable GL_BLEND)
(glDisable GL_CULL_FACE)
```

### Blending

```nano
from "modules/glew/glew.nano" import glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA

(glBlendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA)
```

### Depth

```nano
from "modules/glew/glew.nano" import glDepthFunc, GL_LESS

(glDepthFunc GL_LESS)
```

### Cull Face

```nano
from "modules/glew/glew.nano" import glCullFace, GL_BACK

(glCullFace GL_BACK)
```

## Matrix Stack

```nano
from "modules/glew/glew.nano" import
    glMatrixMode, glLoadIdentity, glOrtho, glFrustum,
    glPushMatrix, glPopMatrix,
    nlg_glTranslatef, nlg_glRotatef, nlg_glScalef,
    GL_MODELVIEW, GL_PROJECTION

# Orthographic (2D) projection
(glMatrixMode GL_PROJECTION)
(glLoadIdentity)
(glOrtho 0.0 800.0 0.0 600.0 -1.0 1.0)

# Perspective projection
(glMatrixMode GL_PROJECTION)
(glLoadIdentity)
(glFrustum -1.0 1.0 -0.75 0.75 0.1 100.0)

# Object transforms
(glMatrixMode GL_MODELVIEW)
(glLoadIdentity)
(glPushMatrix)
    (nlg_glTranslatef 0.5 0.0 -2.0)
    (nlg_glRotatef 30.0 0.0 1.0 0.0)
    (nlg_glScalef 0.5 0.5 0.5)
    # draw object here
(glPopMatrix)
```

## Lighting

```nano
from "modules/glew/glew.nano" import
    glEnable, glShadeModel, glColorMaterial,
    nl_glLightfv4, nl_glMaterialfv4, nlg_glMaterialf,
    GL_LIGHTING, GL_LIGHT0, GL_COLOR_MATERIAL,
    GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_SPECULAR,
    GL_FRONT, GL_SHININESS, GL_SMOOTH

(glEnable GL_LIGHTING)
(glEnable GL_LIGHT0)
(glShadeModel GL_SMOOTH)

# Point light at (3, 5, 3)
(nl_glLightfv4 GL_LIGHT0 GL_POSITION 3.0 5.0 3.0 1.0)
(nl_glLightfv4 GL_LIGHT0 GL_DIFFUSE 1.0 0.95 0.85 1.0)
(nl_glLightfv4 GL_LIGHT0 GL_AMBIENT 0.15 0.15 0.2 1.0)

# Shiny material
(nl_glMaterialfv4 GL_FRONT GL_SPECULAR 1.0 1.0 1.0 1.0)
(nlg_glMaterialf GL_FRONT GL_SHININESS 64.0)

# Allow glColor3f to drive material color
(glEnable GL_COLOR_MATERIAL)
(glColorMaterial GL_FRONT GL_AMBIENT_AND_DIFFUSE)
```

## Modern Pipeline: Shaders and VBOs (nl_gl3_*)

The `nl_gl3_*` helpers bridge NanoLang's type system to the modern OpenGL 3+ pipeline.

### Compiling Shaders

```nano
from "modules/glew/glew.nano" import
    nl_gl3_create_program_from_sources,
    nl_gl3_use_program, nl_gl3_delete_program

let vert: string = "
    #version 330 core
    layout(location = 0) in vec2 pos;
    layout(location = 1) in vec3 col;
    out vec3 vColor;
    void main() {
        gl_Position = vec4(pos, 0.0, 1.0);
        vColor = col;
    }
"

let frag: string = "
    #version 330 core
    in vec3 vColor;
    out vec4 fragColor;
    void main() {
        fragColor = vec4(vColor, 1.0);
    }
"

let prog: int = (nl_gl3_create_program_from_sources vert frag)

# Activate the program for drawing
(nl_gl3_use_program prog)

# Cleanup when done
(nl_gl3_delete_program prog)
```

### Uniform Variables

```nano
from "modules/glew/glew.nano" import
    nl_gl3_get_uniform_location,
    nl_gl3_uniform1f, nl_gl3_uniform2f, nl_gl3_uniform1i

let loc_time: int = (nl_gl3_get_uniform_location prog "uTime")
let loc_res: int = (nl_gl3_get_uniform_location prog "uResolution")

(nl_gl3_uniform1f loc_time 1.23)
(nl_gl3_uniform2f loc_res 800.0 600.0)
(nl_gl3_uniform1i (nl_gl3_get_uniform_location prog "uTexture") 0)
```

### Vertex Buffers (VBOs) and Vertex Arrays (VAOs)

```nano
from "modules/glew/glew.nano" import
    nl_gl3_gen_vertex_array, nl_gl3_bind_vertex_array,
    nl_gl3_gen_buffer, nl_gl3_bind_buffer, nl_gl3_buffer_data_f32,
    nl_gl3_enable_vertex_attrib_array, nl_gl3_vertex_attrib_pointer_f32,
    nl_gl3_draw_arrays,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_TRIANGLES

# Interleaved: [x, y, r, g, b] per vertex
let data: array<float> = [
     0.0,  0.5, 1.0, 0.0, 0.0,   # top, red
    -0.5, -0.5, 0.0, 1.0, 0.0,   # bottom-left, green
     0.5, -0.5, 0.0, 0.0, 1.0    # bottom-right, blue
]

let vao: int = (nl_gl3_gen_vertex_array)
let vbo: int = (nl_gl3_gen_buffer)

(nl_gl3_bind_vertex_array vao)
(nl_gl3_bind_buffer GL_ARRAY_BUFFER vbo)
(nl_gl3_buffer_data_f32 GL_ARRAY_BUFFER data GL_STATIC_DRAW)

# Attribute 0: position (2 floats, stride 20 bytes, offset 0)
(nl_gl3_enable_vertex_attrib_array 0)
(nl_gl3_vertex_attrib_pointer_f32 0 2 0 20 0)

# Attribute 1: color (3 floats, stride 20 bytes, offset 8 bytes)
(nl_gl3_enable_vertex_attrib_array 1)
(nl_gl3_vertex_attrib_pointer_f32 1 3 0 20 8)

# Draw in render loop:
(nl_gl3_use_program prog)
(nl_gl3_bind_vertex_array vao)
(nl_gl3_draw_arrays GL_TRIANGLES 0 3)
```

### Index Buffers (EBOs)

```nano
from "modules/glew/glew.nano" import
    nl_gl3_gen_buffer, nl_gl3_bind_buffer, nl_gl3_buffer_data_u32,
    GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW

let indices: array<int> = [0, 1, 2,  0, 2, 3]   # two triangles = quad
let ebo: int = (nl_gl3_gen_buffer)
(nl_gl3_bind_buffer GL_ELEMENT_ARRAY_BUFFER ebo)
(nl_gl3_buffer_data_u32 GL_ELEMENT_ARRAY_BUFFER indices GL_STATIC_DRAW)
```

### Instanced Drawing

```nano
from "modules/glew/glew.nano" import
    nl_gl3_vertex_attrib_divisor, nl_gl3_draw_arrays_instanced, GL_TRIANGLES

# divisor=1 means this attribute advances once per instance, not per vertex
(nl_gl3_vertex_attrib_divisor 2 1)

# Draw 1000 instances of a 3-vertex triangle
(nl_gl3_draw_arrays_instanced GL_TRIANGLES 0 3 1000)
```

### Textures

```nano
from "modules/glew/glew.nano" import
    nl_gl3_gen_texture, nl_gl3_bind_texture, nl_gl3_active_texture,
    nl_gl3_tex_parami, nl_gl3_tex_image_2d_checker_rgba8,
    GL_TEXTURE_2D, GL_TEXTURE0,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_LINEAR, GL_NEAREST, GL_REPEAT

let tex: int = (nl_gl3_gen_texture)
(nl_gl3_bind_texture GL_TEXTURE_2D tex)
(nl_gl3_tex_parami GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
(nl_gl3_tex_parami GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_NEAREST)
(nl_gl3_tex_parami GL_TEXTURE_2D GL_TEXTURE_WRAP_S GL_REPEAT)
(nl_gl3_tex_parami GL_TEXTURE_2D GL_TEXTURE_WRAP_T GL_REPEAT)

# Fill with a procedural checkerboard for testing
(nl_gl3_tex_image_2d_checker_rgba8 GL_TEXTURE_2D 256 256 8)

# Bind to texture unit 0 when rendering
(nl_gl3_active_texture GL_TEXTURE0)
(nl_gl3_bind_texture GL_TEXTURE_2D tex)
```

### Framebuffer Objects (FBOs)

```nano
from "modules/glew/glew.nano" import
    nl_gl3_gen_framebuffer, nl_gl3_bind_framebuffer,
    nl_gl3_framebuffer_texture_2d, nl_gl3_check_framebuffer_status,
    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, GL_FRAMEBUFFER_COMPLETE

let fbo: int = (nl_gl3_gen_framebuffer)
(nl_gl3_bind_framebuffer GL_FRAMEBUFFER fbo)

# Attach a texture as the color buffer
(nl_gl3_framebuffer_texture_2d GL_FRAMEBUFFER GL_COLOR_ATTACHMENT0 GL_TEXTURE_2D render_tex 0)

let status: int = (nl_gl3_check_framebuffer_status GL_FRAMEBUFFER)
if (!= status GL_FRAMEBUFFER_COMPLETE) {
    (print "Framebuffer not complete")
}

# Render to texture here, then unbind:
(nl_gl3_bind_framebuffer GL_FRAMEBUFFER 0)   # 0 = default framebuffer (screen)
```

## Flush and Finish

```nano
from "modules/glew/glew.nano" import glFlush, glFinish

(glFlush)    # flush commands to GPU (non-blocking)
(glFinish)   # wait until GPU has finished all pending commands (blocking)
```

`glFlush` is rarely needed explicitly; `glfwSwapBuffers` flushes automatically. `glFinish` is useful for benchmarking GPU work.

## API Summary

| Function | Description |
|----------|-------------|
| `glewInit()` | Initialize GLEW, load extensions |
| `glewIsSupported(name)` | 1 if extension is available |
| `glewGetString(name)` | GLEW info string |
| `glewGetErrorString(error)` | Error description |
| `glGetError()` | Get last GL error code |
| `glGetString(name)` | GL info string |
| `glClear(mask)` | Clear buffers |
| `nlg_glClearColor(r, g, b, a)` | Set clear color |
| `glViewport(x, y, w, h)` | Set viewport rectangle |
| `glEnable(cap)` / `glDisable(cap)` | Toggle GL capability |
| `glBlendFunc(src, dst)` | Set blend equation |
| `glDepthFunc(func)` | Set depth test function |
| `glCullFace(mode)` | Set cull face |
| `glMatrixMode(mode)` | Select active matrix |
| `glLoadIdentity()` | Reset to identity matrix |
| `glOrtho(...)` | Set orthographic projection |
| `glFrustum(...)` | Set perspective projection |
| `glPushMatrix()` / `glPopMatrix()` | Save/restore matrix |
| `nlg_glTranslatef(x, y, z)` | Translate matrix |
| `nlg_glRotatef(angle, x, y, z)` | Rotate matrix |
| `nlg_glScalef(x, y, z)` | Scale matrix |
| `glShadeModel(mode)` | Set smooth/flat shading |
| `glColorMaterial(face, mode)` | Track material from glColor |
| `nl_glLightfv4(light, pname, ...)` | Set light parameter (4 floats) |
| `nl_glMaterialfv4(face, pname, ...)` | Set material parameter (4 floats) |
| `nl_gl3_create_program_from_sources(vert, frag)` | Compile GLSL shader program |
| `nl_gl3_use_program(prog)` | Activate shader program |
| `nl_gl3_delete_program(prog)` | Free shader program |
| `nl_gl3_get_uniform_location(prog, name)` | Get uniform location |
| `nl_gl3_uniform1f(loc, v)` | Set float uniform |
| `nl_gl3_uniform2f(loc, x, y)` | Set vec2 uniform |
| `nl_gl3_uniform1i(loc, v)` | Set int/sampler uniform |
| `nl_gl3_gen_vertex_array()` | Create VAO |
| `nl_gl3_bind_vertex_array(vao)` | Bind VAO |
| `nl_gl3_gen_buffer()` | Create VBO/EBO |
| `nl_gl3_bind_buffer(target, buf)` | Bind buffer |
| `nl_gl3_buffer_data_f32(target, data, usage)` | Upload float array to GPU |
| `nl_gl3_buffer_data_u32(target, data, usage)` | Upload int array to GPU |
| `nl_gl3_enable_vertex_attrib_array(idx)` | Enable vertex attribute |
| `nl_gl3_vertex_attrib_pointer_f32(idx, size, norm, stride, offset)` | Describe attribute layout |
| `nl_gl3_draw_arrays(mode, first, count)` | Draw vertices |
| `nl_gl3_draw_arrays_instanced(mode, first, count, instances)` | Draw instanced |
| `nl_gl3_gen_texture()` | Create texture object |
| `nl_gl3_bind_texture(target, tex)` | Bind texture |
| `nl_gl3_active_texture(unit)` | Select texture unit |
| `nl_gl3_tex_parami(target, pname, param)` | Set texture parameter |
| `nl_gl3_gen_framebuffer()` | Create FBO |
| `nl_gl3_bind_framebuffer(target, fbo)` | Bind FBO |
| `nl_gl3_framebuffer_texture_2d(...)` | Attach texture to FBO |
| `nl_gl3_check_framebuffer_status(target)` | Verify FBO completeness |

---

**Previous:** [17.2 GLFW](glfw.html)
**Next:** [17.4 GLUT](glut.html)
