# glfw API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn glfwInit() -> int`

**Returns:** `int`


#### `extern fn glfwTerminate() -> void`

**Returns:** `void`


#### `extern fn glfwCreateWindow(_width: int, _height: int, _title: string, _monitor: GLFWmonitor, _share: GLFWwindow) -> GLFWwindow`

**Parameters:**
| Name | Type |
|------|------|
| `_width` | `int` |
| `_height` | `int` |
| `_title` | `string` |
| `_monitor` | `GLFWmonitor` |
| `_share` | `GLFWwindow` |

**Returns:** `GLFWwindow`


#### `extern fn glfwDestroyWindow(_window: GLFWwindow) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `GLFWwindow` |

**Returns:** `void`


#### `extern fn glfwWindowShouldClose(_window: GLFWwindow) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `GLFWwindow` |

**Returns:** `int`


#### `extern fn glfwSetWindowShouldClose(_window: GLFWwindow, _value: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `GLFWwindow` |
| `_value` | `int` |

**Returns:** `void`


#### `extern fn glfwSwapBuffers(_window: GLFWwindow) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `GLFWwindow` |

**Returns:** `void`


#### `extern fn glfwPollEvents() -> void`

**Returns:** `void`


#### `extern fn glfwMakeContextCurrent(_window: GLFWwindow) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `GLFWwindow` |

**Returns:** `void`


#### `extern fn glfwWindowHint(_hint: int, _value: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_hint` | `int` |
| `_value` | `int` |

**Returns:** `void`


#### `extern fn glfwGetFramebufferSize(_window: GLFWwindow, _width_out: int, _height_out: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `GLFWwindow` |
| `_width_out` | `int` |
| `_height_out` | `int` |

**Returns:** `void`


#### `extern fn glfwSwapInterval(_interval: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_interval` | `int` |

**Returns:** `void`


#### `extern fn glfwGetKey(_window: GLFWwindow, _key: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `GLFWwindow` |
| `_key` | `int` |

**Returns:** `int`


#### `extern fn glfwGetMouseButton(_window: GLFWwindow, _button: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `GLFWwindow` |
| `_button` | `int` |

**Returns:** `int`


#### `extern fn glfwGetCursorPos(_window: GLFWwindow, _xpos_out: int, _ypos_out: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_window` | `GLFWwindow` |
| `_xpos_out` | `int` |
| `_ypos_out` | `int` |

**Returns:** `void`


#### `extern fn glfwGetTime() -> float`

**Returns:** `float`


#### `extern fn glfwSetTime(_time: float) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_time` | `float` |

**Returns:** `void`


### Structs

*No public structs*

### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

- `opaque type GLFWwindow`
- `opaque type GLFWmonitor`

### Constants

| Name | Type | Value |
|------|------|-------|
| `GLFW_PRESS` | `int` | `1` |
| `GLFW_RELEASE` | `int` | `0` |
| `GLFW_REPEAT` | `int` | `2` |
| `GLFW_KEY_SPACE` | `int` | `32` |
| `GLFW_KEY_MINUS` | `int` | `45` |
| `GLFW_KEY_EQUAL` | `int` | `61` |
| `GLFW_KEY_1` | `int` | `49` |
| `GLFW_KEY_2` | `int` | `50` |
| `GLFW_KEY_3` | `int` | `51` |
| `GLFW_KEY_4` | `int` | `52` |
| `GLFW_KEY_5` | `int` | `53` |
| `GLFW_KEY_6` | `int` | `54` |
| `GLFW_KEY_R` | `int` | `82` |
| `GLFW_KEY_ESCAPE` | `int` | `256` |
| `GLFW_KEY_LEFT` | `int` | `263` |
| `GLFW_KEY_RIGHT` | `int` | `262` |
| `GLFW_KEY_DOWN` | `int` | `264` |
| `GLFW_KEY_UP` | `int` | `265` |
| `GLFW_KEY_KP_SUBTRACT` | `int` | `333` |
| `GLFW_KEY_KP_ADD` | `int` | `334` |
| `GLFW_MOUSE_BUTTON_LEFT` | `int` | `0` |
| `GLFW_MOUSE_BUTTON_RIGHT` | `int` | `1` |
| `GLFW_MOUSE_BUTTON_MIDDLE` | `int` | `2` |

