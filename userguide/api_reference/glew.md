# glew API Reference

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


### Structs

*No public structs*

### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

*No opaque types*

### Constants

*No constants*
