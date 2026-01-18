# glfw API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn glfwInit() -> int`

**Returns:** `int`


#### `extern fn glfwTerminate() -> void`

**Returns:** `void`


#### `extern fn glfwCreateWindow(_width: int, _height: int, _title: string, _monitor: struct<GLFWmonitor>, _share: struct<GLFWwindow>) -> struct<GLFWwindow>`

**Parameters:**

| Name | Type |
|------|------|
| `_width` | `int` |
| `_height` | `int` |
| `_title` | `string` |
| `_monitor` | `struct<GLFWmonitor>` |
| `_share` | `struct<GLFWwindow>` |

**Returns:** `struct`


#### `extern fn glfwDestroyWindow(_window: struct<GLFWwindow>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<GLFWwindow>` |

**Returns:** `void`


#### `extern fn glfwWindowShouldClose(_window: struct<GLFWwindow>) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<GLFWwindow>` |

**Returns:** `int`


#### `extern fn glfwSetWindowShouldClose(_window: struct<GLFWwindow>, _value: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<GLFWwindow>` |
| `_value` | `int` |

**Returns:** `void`


#### `extern fn glfwSwapBuffers(_window: struct<GLFWwindow>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<GLFWwindow>` |

**Returns:** `void`


#### `extern fn glfwPollEvents() -> void`

**Returns:** `void`


#### `extern fn glfwMakeContextCurrent(_window: struct<GLFWwindow>) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<GLFWwindow>` |

**Returns:** `void`


#### `extern fn glfwWindowHint(_hint: int, _value: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_hint` | `int` |
| `_value` | `int` |

**Returns:** `void`


#### `extern fn glfwGetFramebufferSize(_window: struct<GLFWwindow>, _width_out: int, _height_out: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<GLFWwindow>` |
| `_width_out` | `int` |
| `_height_out` | `int` |

**Returns:** `void`


#### `extern fn glfwSwapInterval(_interval: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_interval` | `int` |

**Returns:** `void`


#### `extern fn glfwGetKey(_window: struct<GLFWwindow>, _key: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<GLFWwindow>` |
| `_key` | `int` |

**Returns:** `int`


#### `extern fn glfwGetMouseButton(_window: struct<GLFWwindow>, _button: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<GLFWwindow>` |
| `_button` | `int` |

**Returns:** `int`


#### `extern fn glfwGetCursorPos(_window: struct<GLFWwindow>, _xpos_out: int, _ypos_out: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_window` | `struct<GLFWwindow>` |
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

*No constants*
