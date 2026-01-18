# bullet API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn nl_bullet_init() -> int`

**Returns:** `int`


#### `extern fn nl_bullet_cleanup() -> void`

**Returns:** `void`


#### `extern fn nl_bullet_step(_time_step: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_time_step` | `float` |

**Returns:** `void`


#### `extern fn nl_bullet_set_gravity(_gx: float, _gy: float, _gz: float) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_gx` | `float` |
| `_gy` | `float` |
| `_gz` | `float` |

**Returns:** `void`


#### `extern fn nl_bullet_create_soft_sphere(_x: float, _y: float, _z: float, _radius: float, _resolution: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_x` | `float` |
| `_y` | `float` |
| `_z` | `float` |
| `_radius` | `float` |
| `_resolution` | `int` |

**Returns:** `int`


#### `extern fn nl_bullet_create_rigid_sphere(_x: float, _y: float, _z: float, _radius: float, _mass: float, _restitution: float) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_x` | `float` |
| `_y` | `float` |
| `_z` | `float` |
| `_radius` | `float` |
| `_mass` | `float` |
| `_restitution` | `float` |

**Returns:** `int`


#### `extern fn nl_bullet_create_rigid_box(_x: float, _y: float, _z: float, _half_width: float, _half_height: float, _half_depth: float, _mass: float, _restitution: float) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_x` | `float` |
| `_y` | `float` |
| `_z` | `float` |
| `_half_width` | `float` |
| `_half_height` | `float` |
| `_half_depth` | `float` |
| `_mass` | `float` |
| `_restitution` | `float` |

**Returns:** `int`


#### `extern fn nl_bullet_create_rigid_box_rotated(_x: float, _y: float, _z: float, _half_width: float, _half_height: float, _half_depth: float, _angle_degrees: float, _mass: float, _restitution: float) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_x` | `float` |
| `_y` | `float` |
| `_z` | `float` |
| `_half_width` | `float` |
| `_half_height` | `float` |
| `_half_depth` | `float` |
| `_angle_degrees` | `float` |
| `_mass` | `float` |
| `_restitution` | `float` |

**Returns:** `int`


#### `extern fn nl_bullet_get_soft_body_count() -> int`

**Returns:** `int`


#### `extern fn nl_bullet_get_soft_body_node_count(_handle: int) -> int`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `int`


#### `extern fn nl_bullet_get_soft_body_node_x(_handle: int, _node_idx: int) -> float`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |
| `_node_idx` | `int` |

**Returns:** `float`


#### `extern fn nl_bullet_get_soft_body_node_y(_handle: int, _node_idx: int) -> float`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |
| `_node_idx` | `int` |

**Returns:** `float`


#### `extern fn nl_bullet_get_soft_body_node_z(_handle: int, _node_idx: int) -> float`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |
| `_node_idx` | `int` |

**Returns:** `float`


#### `extern fn nl_bullet_remove_soft_body(_handle: int) -> void`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `void`


#### `extern fn nl_bullet_get_rigid_body_count() -> int`

**Returns:** `int`


#### `extern fn nl_bullet_get_rigid_body_x(_handle: int) -> float`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `float`


#### `extern fn nl_bullet_get_rigid_body_y(_handle: int) -> float`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `float`


#### `extern fn nl_bullet_get_rigid_body_z(_handle: int) -> float`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `float`


#### `extern fn nl_bullet_get_rigid_body_rot_x(_handle: int) -> float`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `float`


#### `extern fn nl_bullet_get_rigid_body_rot_y(_handle: int) -> float`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `float`


#### `extern fn nl_bullet_get_rigid_body_rot_z(_handle: int) -> float`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `float`


#### `extern fn nl_bullet_get_rigid_body_rot_w(_handle: int) -> float`

**Parameters:**

| Name | Type |
|------|------|
| `_handle` | `int` |

**Returns:** `float`


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
