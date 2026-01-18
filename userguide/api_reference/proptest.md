# proptest API Reference

*Auto-generated from module reflection*


### Functions

#### `fn prop_pass() -> string`

**Returns:** `string`


#### `fn prop_fail(message: string) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `message` | `string` |

**Returns:** `string`


#### `fn prop_discard(reason: string) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `reason` | `string` |

**Returns:** `string`


#### `fn config_default() -> struct<RunConfig>`

**Returns:** `struct`


#### `fn config(trials: int, max_shrink_steps: int, discard_limit: int, seed: int) -> struct<RunConfig>`

**Parameters:**

| Name | Type |
|------|------|
| `trials` | `int` |
| `max_shrink_steps` | `int` |
| `discard_limit` | `int` |
| `seed` | `int` |

**Returns:** `struct`


#### `fn int_range(min_value: int, max_value: int) -> struct<IntRangeGenerator>`

**Parameters:**

| Name | Type |
|------|------|
| `min_value` | `int` |
| `max_value` | `int` |

**Returns:** `struct`


#### `fn int_pair(first: struct<IntRangeGenerator>, second: struct<IntRangeGenerator>) -> struct<IntPairGenerator>`

**Parameters:**

| Name | Type |
|------|------|
| `first` | `struct<IntRangeGenerator>` |
| `second` | `struct<IntRangeGenerator>` |

**Returns:** `struct`


#### `fn int_array(element: struct<IntRangeGenerator>, max_length: int) -> struct<IntArrayGenerator>`

**Parameters:**

| Name | Type |
|------|------|
| `element` | `struct<IntRangeGenerator>` |
| `max_length` | `int` |

**Returns:** `struct`


#### `fn report_passed(report: struct<PropertyReport>) -> bool`

**Parameters:**

| Name | Type |
|------|------|
| `report` | `struct<PropertyReport>` |

**Returns:** `bool`


#### `fn report_summary(report: struct<PropertyReport>) -> string`

**Parameters:**

| Name | Type |
|------|------|
| `report` | `struct<PropertyReport>` |

**Returns:** `string`


#### `fn forall_int(name: string, generator: struct<IntRangeGenerator>, property: unknown) -> struct<PropertyReport>`

**Parameters:**

| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `struct<IntRangeGenerator>` |
| `property` | `unknown` |

**Returns:** `struct`


#### `fn forall_int_with_config(name: string, generator: struct<IntRangeGenerator>, property: unknown, cfg: struct<RunConfig>) -> struct<PropertyReport>`

**Parameters:**

| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `struct<IntRangeGenerator>` |
| `property` | `unknown` |
| `cfg` | `struct<RunConfig>` |

**Returns:** `struct`


#### `fn forall_int_pair(name: string, generator: struct<IntPairGenerator>, property: unknown) -> struct<PropertyReport>`

**Parameters:**

| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `struct<IntPairGenerator>` |
| `property` | `unknown` |

**Returns:** `struct`


#### `fn forall_int_pair_with_config(name: string, generator: struct<IntPairGenerator>, property: unknown, cfg: struct<RunConfig>) -> struct<PropertyReport>`

**Parameters:**

| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `struct<IntPairGenerator>` |
| `property` | `unknown` |
| `cfg` | `struct<RunConfig>` |

**Returns:** `struct`


#### `fn forall_int_array(name: string, generator: struct<IntArrayGenerator>, property: unknown) -> struct<PropertyReport>`

**Parameters:**

| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `struct<IntArrayGenerator>` |
| `property` | `unknown` |

**Returns:** `struct`


#### `fn forall_int_array_with_config(name: string, generator: struct<IntArrayGenerator>, property: unknown, cfg: struct<RunConfig>) -> struct<PropertyReport>`

**Parameters:**

| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `struct<IntArrayGenerator>` |
| `property` | `unknown` |
| `cfg` | `struct<RunConfig>` |

**Returns:** `struct`


### Structs

#### `struct RunConfig`

**Fields:**

| Name | Type |
|------|------|
| `trials` | `int` |
| `max_shrink_steps` | `int` |
| `discard_limit` | `int` |
| `seed` | `int` |


#### `struct PropertyReport`

**Fields:**

| Name | Type |
|------|------|
| `name` | `string` |
| `passed` | `bool` |
| `case_count` | `int` |
| `discard_count` | `int` |
| `shrink_count` | `int` |
| `counterexample` | `string` |


#### `struct IntRangeGenerator`

**Fields:**

| Name | Type |
|------|------|
| `min` | `int` |
| `max` | `int` |


#### `struct IntPairGenerator`

**Fields:**

| Name | Type |
|------|------|
| `first` | `struct<IntRangeGenerator>` |
| `second` | `struct<IntRangeGenerator>` |


#### `struct IntArrayGenerator`

**Fields:**

| Name | Type |
|------|------|
| `element` | `struct<IntRangeGenerator>` |
| `max_length` | `int` |


### Enums

*No public enums*

### Unions

*No public unions*

### Opaque Types

*No opaque types*

### Constants

*No constants*
