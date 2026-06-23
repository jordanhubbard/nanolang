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


#### `fn config_default() -> RunConfig`

**Returns:** `RunConfig`


#### `fn config(trials: int, max_shrink_steps: int, discard_limit: int, seed: int) -> RunConfig`

**Parameters:**
| Name | Type |
|------|------|
| `trials` | `int` |
| `max_shrink_steps` | `int` |
| `discard_limit` | `int` |
| `seed` | `int` |

**Returns:** `RunConfig`


#### `fn int_range(min_value: int, max_value: int) -> IntRangeGenerator`

**Parameters:**
| Name | Type |
|------|------|
| `min_value` | `int` |
| `max_value` | `int` |

**Returns:** `IntRangeGenerator`


#### `fn int_pair(first: IntRangeGenerator, second: IntRangeGenerator) -> IntPairGenerator`

**Parameters:**
| Name | Type |
|------|------|
| `first` | `IntRangeGenerator` |
| `second` | `IntRangeGenerator` |

**Returns:** `IntPairGenerator`


#### `fn int_array(element: IntRangeGenerator, max_length: int) -> IntArrayGenerator`

**Parameters:**
| Name | Type |
|------|------|
| `element` | `IntRangeGenerator` |
| `max_length` | `int` |

**Returns:** `IntArrayGenerator`


#### `fn report_passed(report: PropertyReport) -> bool`

**Parameters:**
| Name | Type |
|------|------|
| `report` | `PropertyReport` |

**Returns:** `bool`


#### `fn report_summary(report: PropertyReport) -> string`

**Parameters:**
| Name | Type |
|------|------|
| `report` | `PropertyReport` |

**Returns:** `string`


#### `fn forall_int(name: string, generator: IntRangeGenerator, property: unknown) -> PropertyReport`

**Parameters:**
| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `IntRangeGenerator` |
| `property` | `unknown` |

**Returns:** `PropertyReport`


#### `fn forall_int_with_config(name: string, generator: IntRangeGenerator, property: unknown, cfg: RunConfig) -> PropertyReport`

**Parameters:**
| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `IntRangeGenerator` |
| `property` | `unknown` |
| `cfg` | `RunConfig` |

**Returns:** `PropertyReport`


#### `fn forall_int_pair(name: string, generator: IntPairGenerator, property: unknown) -> PropertyReport`

**Parameters:**
| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `IntPairGenerator` |
| `property` | `unknown` |

**Returns:** `PropertyReport`


#### `fn forall_int_pair_with_config(name: string, generator: IntPairGenerator, property: unknown, cfg: RunConfig) -> PropertyReport`

**Parameters:**
| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `IntPairGenerator` |
| `property` | `unknown` |
| `cfg` | `RunConfig` |

**Returns:** `PropertyReport`


#### `fn forall_int_array(name: string, generator: IntArrayGenerator, property: unknown) -> PropertyReport`

**Parameters:**
| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `IntArrayGenerator` |
| `property` | `unknown` |

**Returns:** `PropertyReport`


#### `fn forall_int_array_with_config(name: string, generator: IntArrayGenerator, property: unknown, cfg: RunConfig) -> PropertyReport`

**Parameters:**
| Name | Type |
|------|------|
| `name` | `string` |
| `generator` | `IntArrayGenerator` |
| `property` | `unknown` |
| `cfg` | `RunConfig` |

**Returns:** `PropertyReport`


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
| `PROP_OUTCOME_PASS` | `int` | `0` |
| `PROP_OUTCOME_FAIL` | `int` | `1` |
| `PROP_OUTCOME_DISCARD` | `int` | `2` |
| `RNG_MULTIPLIER` | `int` | `48271` |
| `RNG_MODULUS` | `int` | `2147483647` |

