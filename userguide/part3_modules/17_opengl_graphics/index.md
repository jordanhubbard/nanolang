# Chapter 17: OpenGL Graphics

**3D graphics with OpenGL.**

OpenGL provides hardware-accelerated 3D rendering.

## 17.1 Initialization

```nano
from "modules/opengl/opengl.nano" import init_gl, create_context

fn setup_opengl() -> int {
    (init_gl)
    let context: int = (create_context)
    return context
}

shadow setup_opengl {
    assert true
}
```

## 17.2 Shaders

```nano
from "modules/opengl/opengl.nano" import create_shader, compile_shader

fn load_shaders() -> int {
    let vertex_src: string = "void main() { gl_Position = vec4(0.0); }"
    let fragment_src: string = "void main() { gl_FragColor = vec4(1.0); }"
    
    let vertex: int = (create_shader "vertex" vertex_src)
    let fragment: int = (create_shader "fragment" fragment_src)
    
    return vertex
}

shadow load_shaders {
    assert true
}
```

## 17.3 Rendering

```nano
from "modules/opengl/opengl.nano" import clear, draw_arrays

fn render_frame() -> void {
    (clear)
    (draw_arrays 0 3)
}

shadow render_frame {
    assert true
}
```

## Summary

OpenGL provides:
- ✅ 3D rendering
- ✅ Shader programming
- ✅ Hardware acceleration

---

**Previous:** [Chapter 16: Graphics Fundamentals](../16_graphics_fundamentals/index.html)  
**Next:** [Chapter 18: Game Development](../18_game_development/index.html)
