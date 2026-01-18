# opengl API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn glewInit() -> int`

**Returns:** `int`


#### `extern fn glewIsSupported(_extension: string) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_extension` | `string` |

**Returns:** `int`


#### `extern fn glewGetString(_name: int) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_name` | `int` |

**Returns:** `string`


#### `extern fn glewGetErrorString(_error: int) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_error` | `int` |

**Returns:** `string`


#### `extern fn glGetError() -> int`

**Returns:** `int`


#### `extern fn glGetString(_name: int) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `_name` | `int` |

**Returns:** `string`


#### `extern fn glClear(_mask: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_mask` | `int` |

**Returns:** `void`


#### `extern fn nlg_glClearColor(_r: float, _g: float, _b: float, _a: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_r` | `float` |
| `_g` | `float` |
| `_b` | `float` |
| `_a` | `float` |

**Returns:** `void`


#### `extern fn glViewport(_x: int, _y: int, _width: int, _height: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_x` | `int` |
| `_y` | `int` |
| `_width` | `int` |
| `_height` | `int` |

**Returns:** `void`


#### `extern fn glFlush() -> void`

**Returns:** `void`


#### `extern fn glFinish() -> void`

**Returns:** `void`


#### `extern fn glBegin(_mode: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_mode` | `int` |

**Returns:** `void`


#### `extern fn glEnd() -> void`

**Returns:** `void`


#### `extern fn nlg_glVertex2f(_x: float, _y: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_x` | `float` |
| `_y` | `float` |

**Returns:** `void`


#### `extern fn nlg_glVertex3f(_x: float, _y: float, _z: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_x` | `float` |
| `_y` | `float` |
| `_z` | `float` |

**Returns:** `void`


#### `extern fn nlg_glColor3f(_r: float, _g: float, _b: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_r` | `float` |
| `_g` | `float` |
| `_b` | `float` |

**Returns:** `void`


#### `extern fn nlg_glColor4f(_r: float, _g: float, _b: float, _a: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_r` | `float` |
| `_g` | `float` |
| `_b` | `float` |
| `_a` | `float` |

**Returns:** `void`


#### `extern fn glMatrixMode(_mode: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_mode` | `int` |

**Returns:** `void`


#### `extern fn glLoadIdentity() -> void`

**Returns:** `void`


#### `extern fn glOrtho(_left: float, _right: float, _bottom: float, _top: float, _near: float, _far: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_left` | `float` |
| `_right` | `float` |
| `_bottom` | `float` |
| `_top` | `float` |
| `_near` | `float` |
| `_far` | `float` |

**Returns:** `void`


#### `extern fn glFrustum(_left: float, _right: float, _bottom: float, _top: float, _near: float, _far: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_left` | `float` |
| `_right` | `float` |
| `_bottom` | `float` |
| `_top` | `float` |
| `_near` | `float` |
| `_far` | `float` |

**Returns:** `void`


#### `extern fn nlg_glTranslatef(_x: float, _y: float, _z: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_x` | `float` |
| `_y` | `float` |
| `_z` | `float` |

**Returns:** `void`


#### `extern fn nlg_glRotatef(_angle: float, _x: float, _y: float, _z: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_angle` | `float` |
| `_x` | `float` |
| `_y` | `float` |
| `_z` | `float` |

**Returns:** `void`


#### `extern fn nlg_glScalef(_x: float, _y: float, _z: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_x` | `float` |
| `_y` | `float` |
| `_z` | `float` |

**Returns:** `void`


#### `extern fn glShadeModel(_mode: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_mode` | `int` |

**Returns:** `void`


#### `extern fn glColorMaterial(_face: int, _mode: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_face` | `int` |
| `_mode` | `int` |

**Returns:** `void`


#### `extern fn glEnable(_cap: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_cap` | `int` |

**Returns:** `void`


#### `extern fn glDisable(_cap: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_cap` | `int` |

**Returns:** `void`


#### `extern fn glPushAttrib(_mask: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_mask` | `int` |

**Returns:** `void`


#### `extern fn glPopAttrib() -> void`

**Returns:** `void`


#### `extern fn glPushMatrix() -> void`

**Returns:** `void`


#### `extern fn glPopMatrix() -> void`

**Returns:** `void`


#### `extern fn glPolygonMode(_face: int, _mode: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_face` | `int` |
| `_mode` | `int` |

**Returns:** `void`


#### `extern fn nlg_glLineWidth(_width: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_width` | `float` |

**Returns:** `void`


#### `extern fn nlg_glPointSize(_size: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_size` | `float` |

**Returns:** `void`


#### `extern fn glBlendFunc(_sfactor: int, _dfactor: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_sfactor` | `int` |
| `_dfactor` | `int` |

**Returns:** `void`


#### `extern fn nlg_glNormal3f(_nx: float, _ny: float, _nz: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_nx` | `float` |
| `_ny` | `float` |
| `_nz` | `float` |

**Returns:** `void`


#### `extern fn nl_glLightfv4(_light: int, _pname: int, _x: float, _y: float, _z: float, _w: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_light` | `int` |
| `_pname` | `int` |
| `_x` | `float` |
| `_y` | `float` |
| `_z` | `float` |
| `_w` | `float` |

**Returns:** `void`


#### `extern fn nl_glMaterialfv4(_face: int, _pname: int, _x: float, _y: float, _z: float, _w: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_face` | `int` |
| `_pname` | `int` |
| `_x` | `float` |
| `_y` | `float` |
| `_z` | `float` |
| `_w` | `float` |

**Returns:** `void`


#### `extern fn nlg_glMaterialf(_face: int, _pname: int, _param: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_face` | `int` |
| `_pname` | `int` |
| `_param` | `float` |

**Returns:** `void`


#### `extern fn nl_gl3_create_program_from_sources(_vertex_src: string, _fragment_src: string) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_vertex_src` | `string` |
| `_fragment_src` | `string` |

**Returns:** `int`


#### `extern fn nl_gl3_use_program(_program: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_program` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_delete_program(_program: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_program` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_get_uniform_location(_program: int, _name: string) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_program` | `int` |
| `_name` | `string` |

**Returns:** `int`


#### `extern fn nl_gl3_uniform1f(_location: int, _v: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_location` | `int` |
| `_v` | `float` |

**Returns:** `void`


#### `extern fn nl_gl3_uniform2f(_location: int, _x: float, _y: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_location` | `int` |
| `_x` | `float` |
| `_y` | `float` |

**Returns:** `void`


#### `extern fn nl_gl3_uniform1i(_location: int, _v: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_location` | `int` |
| `_v` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_gen_vertex_array() -> int`

**Returns:** `int`


#### `extern fn nl_gl3_bind_vertex_array(_vao: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_vao` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_gen_buffer() -> int`

**Returns:** `int`


#### `extern fn nl_gl3_bind_buffer(_target: int, _buffer: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_target` | `int` |
| `_buffer` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_buffer_data_f32(_target: int, _data: array, _usage: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_target` | `int` |
| `_data` | `array` |
| `_usage` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_buffer_data_u32(_target: int, _data: array, _usage: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_target` | `int` |
| `_data` | `array` |
| `_usage` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_enable_vertex_attrib_array(_index: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_index` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_vertex_attrib_pointer_f32(_index: int, _size: int, _normalized: int, _stride_bytes: int, _offset_bytes: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_index` | `int` |
| `_size` | `int` |
| `_normalized` | `int` |
| `_stride_bytes` | `int` |
| `_offset_bytes` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_vertex_attrib_divisor(_index: int, _divisor: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_index` | `int` |
| `_divisor` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_draw_arrays(_mode: int, _first: int, _count: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_mode` | `int` |
| `_first` | `int` |
| `_count` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_draw_arrays_instanced(_mode: int, _first: int, _count: int, _instance_count: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_mode` | `int` |
| `_first` | `int` |
| `_count` | `int` |
| `_instance_count` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_gen_texture() -> int`

**Returns:** `int`


#### `extern fn nl_gl3_bind_texture(_target: int, _texture: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_target` | `int` |
| `_texture` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_active_texture(_texture_unit: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_texture_unit` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_tex_parami(_target: int, _pname: int, _param: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_target` | `int` |
| `_pname` | `int` |
| `_param` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_tex_image_2d_checker_rgba8(_target: int, _width: int, _height: int, _squares: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_target` | `int` |
| `_width` | `int` |
| `_height` | `int` |
| `_squares` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_gen_framebuffer() -> int`

**Returns:** `int`


#### `extern fn nl_gl3_bind_framebuffer(_target: int, _fbo: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_target` | `int` |
| `_fbo` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_framebuffer_texture_2d(_target: int, _attachment: int, _textarget: int, _texture: int, _level: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_target` | `int` |
| `_attachment` | `int` |
| `_textarget` | `int` |
| `_texture` | `int` |
| `_level` | `int` |

**Returns:** `void`


#### `extern fn nl_gl3_check_framebuffer_status(_target: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_target` | `int` |

**Returns:** `int`


#### `extern fn glDepthFunc(_func: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_func` | `int` |

**Returns:** `void`


#### `extern fn glCullFace(_mode: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_mode` | `int` |

**Returns:** `void`


#### `extern fn nlg_glRasterPos2f(_x: float, _y: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_x` | `float` |
| `_y` | `float` |

**Returns:** `void`


#### `fn glClearColor(r: float, g: float, b: float, a: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `r` | `float` |
| `g` | `float` |
| `b` | `float` |
| `a` | `float` |

**Returns:** `void`


#### `fn glVertex2f(x: float, y: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `x` | `float` |
| `y` | `float` |

**Returns:** `void`


#### `fn glVertex3f(x: float, y: float, z: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `x` | `float` |
| `y` | `float` |
| `z` | `float` |

**Returns:** `void`


#### `fn glColor3f(r: float, g: float, b: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `r` | `float` |
| `g` | `float` |
| `b` | `float` |

**Returns:** `void`


#### `fn glColor4f(r: float, g: float, b: float, a: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `r` | `float` |
| `g` | `float` |
| `b` | `float` |
| `a` | `float` |

**Returns:** `void`


#### `fn glTranslatef(x: float, y: float, z: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `x` | `float` |
| `y` | `float` |
| `z` | `float` |

**Returns:** `void`


#### `fn glRotatef(angle: float, x: float, y: float, z: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `angle` | `float` |
| `x` | `float` |
| `y` | `float` |
| `z` | `float` |

**Returns:** `void`


#### `fn glScalef(x: float, y: float, z: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `x` | `float` |
| `y` | `float` |
| `z` | `float` |

**Returns:** `void`


#### `fn glLineWidth(width: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `width` | `float` |

**Returns:** `void`


#### `fn glPointSize(size: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `size` | `float` |

**Returns:** `void`


#### `fn glNormal3f(nx: float, ny: float, nz: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `nx` | `float` |
| `ny` | `float` |
| `nz` | `float` |

**Returns:** `void`


#### `fn glRasterPos2f(x: float, y: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `x` | `float` |
| `y` | `float` |

**Returns:** `void`


#### `fn glMaterialf(face: int, pname: int, param: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `face` | `int` |
| `pname` | `int` |
| `param` | `float` |

**Returns:** `void`


### Structs

*No public structs*

### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

*No opaque types*

### Constants

| Name | Type | Value |
|------|------|-------|
| `GLEW_OK` | `int` | `0` |
| `GL_NO_ERROR` | `int` | `0` |
| `GL_ARRAY_BUFFER` | `int` | `34962` |
| `GL_ELEMENT_ARRAY_BUFFER` | `int` | `34963` |
| `GL_STATIC_DRAW` | `int` | `35044` |
| `GL_DYNAMIC_DRAW` | `int` | `35048` |
| `GL_FLOAT` | `int` | `5126` |
| `GL_VERTEX_SHADER` | `int` | `35633` |
| `GL_FRAGMENT_SHADER` | `int` | `35632` |
| `GL_COMPILE_STATUS` | `int` | `35713` |
| `GL_LINK_STATUS` | `int` | `35714` |
| `GL_INFO_LOG_LENGTH` | `int` | `35716` |
| `GL_TEXTURE_2D` | `int` | `3553` |
| `GL_TEXTURE0` | `int` | `33984` |
| `GL_TEXTURE_MIN_FILTER` | `int` | `10241` |
| `GL_TEXTURE_MAG_FILTER` | `int` | `10240` |
| `GL_TEXTURE_WRAP_S` | `int` | `10242` |
| `GL_TEXTURE_WRAP_T` | `int` | `10243` |
| `GL_LINEAR` | `int` | `9729` |
| `GL_NEAREST` | `int` | `9728` |
| `GL_REPEAT` | `int` | `10497` |
| `GL_FRAMEBUFFER` | `int` | `36160` |
| `GL_COLOR_ATTACHMENT0` | `int` | `36064` |
| `GL_FRAMEBUFFER_COMPLETE` | `int` | `36053` |
| `GL_DEPTH_TEST` | `int` | `2929` |
| `GL_LIGHTING` | `int` | `2896` |
| `GL_LIGHT0` | `int` | `16384` |
| `GL_COLOR_MATERIAL` | `int` | `2903` |
| `GL_BLEND` | `int` | `3042` |
| `GL_CULL_FACE` | `int` | `2884` |
| `GL_NORMALIZE` | `int` | `2977` |
| `GL_MODELVIEW` | `int` | `5888` |
| `GL_PROJECTION` | `int` | `5889` |
| `GL_COLOR_BUFFER_BIT` | `int` | `16384` |
| `GL_DEPTH_BUFFER_BIT` | `int` | `256` |
| `GL_POINTS` | `int` | `0` |
| `GL_LINES` | `int` | `1` |
| `GL_LINE_LOOP` | `int` | `2` |
| `GL_LINE_STRIP` | `int` | `3` |
| `GL_TRIANGLES` | `int` | `4` |
| `GL_TRIANGLE_STRIP` | `int` | `5` |
| `GL_TRIANGLE_FAN` | `int` | `6` |
| `GL_QUADS` | `int` | `7` |
| `GL_QUAD_STRIP` | `int` | `8` |
| `GL_POLYGON` | `int` | `9` |
| `GL_LINE` | `int` | `6913` |
| `GL_FILL` | `int` | `6914` |
| `GL_POSITION` | `int` | `4611` |
| `GL_AMBIENT` | `int` | `4608` |
| `GL_DIFFUSE` | `int` | `4609` |
| `GL_SPECULAR` | `int` | `4610` |
| `GL_FRONT` | `int` | `1028` |
| `GL_BACK` | `int` | `1029` |
| `GL_FRONT_AND_BACK` | `int` | `1032` |
| `GL_AMBIENT_AND_DIFFUSE` | `int` | `5634` |
| `GL_SHININESS` | `int` | `5633` |
| `GL_SMOOTH` | `int` | `7425` |
| `GL_FLAT` | `int` | `7424` |
| `GL_SRC_ALPHA` | `int` | `770` |
| `GL_ONE_MINUS_SRC_ALPHA` | `int` | `771` |
| `GL_LESS` | `int` | `513` |
| `GL_ENABLE_BIT` | `int` | `8192` |
