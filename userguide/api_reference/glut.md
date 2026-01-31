# glut API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn glutInit(_argcp: int, _argv: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_argcp` | `int` |
| `_argv` | `int` |

**Returns:** `void`


#### `extern fn glutInitDisplayMode(_mode: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_mode` | `int` |

**Returns:** `void`


#### `extern fn glutInitWindowSize(_width: int, _height: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_width` | `int` |
| `_height` | `int` |

**Returns:** `void`


#### `extern fn glutInitWindowPosition(_x: int, _y: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_x` | `int` |
| `_y` | `int` |

**Returns:** `void`


#### `extern fn glutCreateWindow(_title: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_title` | `string` |

**Returns:** `int`


#### `extern fn glutSolidTeapot(_size: float) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_size` | `float` |

**Returns:** `void`


#### `extern fn glutWireTeapot(_size: float) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_size` | `float` |

**Returns:** `void`


#### `extern fn glutSolidSphere(_radius: float, _slices: int, _stacks: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_radius` | `float` |
| `_slices` | `int` |
| `_stacks` | `int` |

**Returns:** `void`


#### `extern fn glutWireSphere(_radius: float, _slices: int, _stacks: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_radius` | `float` |
| `_slices` | `int` |
| `_stacks` | `int` |

**Returns:** `void`


#### `extern fn glutSolidCube(_size: float) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_size` | `float` |

**Returns:** `void`


#### `extern fn glutWireCube(_size: float) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_size` | `float` |

**Returns:** `void`


#### `extern fn glutSolidCone(_base: float, _height: float, _slices: int, _stacks: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `float` |
| `_height` | `float` |
| `_slices` | `int` |
| `_stacks` | `int` |

**Returns:** `void`


#### `extern fn glutWireCone(_base: float, _height: float, _slices: int, _stacks: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_base` | `float` |
| `_height` | `float` |
| `_slices` | `int` |
| `_stacks` | `int` |

**Returns:** `void`


#### `extern fn glutSolidTorus(_innerRadius: float, _outerRadius: float, _sides: int, _rings: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_innerRadius` | `float` |
| `_outerRadius` | `float` |
| `_sides` | `int` |
| `_rings` | `int` |

**Returns:** `void`


#### `extern fn glutWireTorus(_innerRadius: float, _outerRadius: float, _sides: int, _rings: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_innerRadius` | `float` |
| `_outerRadius` | `float` |
| `_sides` | `int` |
| `_rings` | `int` |

**Returns:** `void`


#### `extern fn glutSolidDodecahedron() -> void`

**Returns:** `void`


#### `extern fn glutWireDodecahedron() -> void`

**Returns:** `void`


#### `extern fn glutSolidOctahedron() -> void`

**Returns:** `void`


#### `extern fn glutWireOctahedron() -> void`

**Returns:** `void`


#### `extern fn glutSolidTetrahedron() -> void`

**Returns:** `void`


#### `extern fn glutWireTetrahedron() -> void`

**Returns:** `void`


#### `extern fn glutSolidIcosahedron() -> void`

**Returns:** `void`


#### `extern fn glutWireIcosahedron() -> void`

**Returns:** `void`


#### `extern fn glutBitmapCharacter(_font: int, _character: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_font` | `int` |
| `_character` | `int` |

**Returns:** `void`


#### `extern fn glutBitmapWidth(_font: int, _character: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_font` | `int` |
| `_character` | `int` |

**Returns:** `int`


#### `extern fn glutBitmapLength(_font: int, _str: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_font` | `int` |
| `_str` | `string` |

**Returns:** `int`


#### `extern fn glutGet(_state: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_state` | `int` |

**Returns:** `int`


#### `extern fn glutSwapBuffers() -> void`

**Returns:** `void`


#### `extern fn glutPostRedisplay() -> void`

**Returns:** `void`


#### `extern fn glutMainLoop() -> void`

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
| `GLUT_RGB` | `int` | `0` |
| `GLUT_RGBA` | `int` | `0` |
| `GLUT_INDEX` | `int` | `1` |
| `GLUT_SINGLE` | `int` | `0` |
| `GLUT_DOUBLE` | `int` | `2` |
| `GLUT_ACCUM` | `int` | `4` |
| `GLUT_ALPHA` | `int` | `8` |
| `GLUT_DEPTH` | `int` | `16` |
| `GLUT_STENCIL` | `int` | `32` |
| `GLUT_POINT` | `int` | `0` |
| `GLUT_LINE` | `int` | `1` |
| `GLUT_FILL` | `int` | `2` |
| `GLUT_BITMAP_9_BY_15` | `int` | `2` |
| `GLUT_BITMAP_8_BY_13` | `int` | `3` |
| `GLUT_BITMAP_TIMES_ROMAN_10` | `int` | `4` |
| `GLUT_BITMAP_TIMES_ROMAN_24` | `int` | `5` |
| `GLUT_BITMAP_HELVETICA_10` | `int` | `6` |
| `GLUT_BITMAP_HELVETICA_12` | `int` | `7` |
| `GLUT_BITMAP_HELVETICA_18` | `int` | `8` |

