# ncurses API Reference

*Auto-generated from module reflection*


### Functions

#### `extern fn initscr_wrapper() -> int`

**Returns:** `int`


#### `extern fn endwin_wrapper() -> int`

**Returns:** `int`


#### `extern fn curs_set_wrapper(_visibility: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_visibility` | `int` |

**Returns:** `int`


#### `extern fn clear_wrapper() -> int`

**Returns:** `int`


#### `extern fn refresh_wrapper() -> int`

**Returns:** `int`


#### `extern fn erase_wrapper() -> int`

**Returns:** `int`


#### `extern fn move_wrapper(_y: int, _x: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_y` | `int` |
| `_x` | `int` |

**Returns:** `int`


#### `extern fn mvprintw_wrapper(_y: int, _x: int, _str: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_y` | `int` |
| `_x` | `int` |
| `_str` | `string` |

**Returns:** `int`


#### `extern fn mvaddch_wrapper(_y: int, _x: int, _ch: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_y` | `int` |
| `_x` | `int` |
| `_ch` | `int` |

**Returns:** `int`


#### `extern fn mvaddstr_wrapper(_y: int, _x: int, _str: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_y` | `int` |
| `_x` | `int` |
| `_str` | `string` |

**Returns:** `int`


#### `extern fn addch_wrapper(_ch: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_ch` | `int` |

**Returns:** `int`


#### `extern fn addstr_wrapper(_str: string) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_str` | `string` |

**Returns:** `int`


#### `extern fn getch_wrapper() -> int`

**Returns:** `int`


#### `extern fn nl_nodelay(_win: int, _bf: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_win` | `int` |
| `_bf` | `int` |

**Returns:** `int`


#### `extern fn nl_keypad(_win: int, _bf: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_win` | `int` |
| `_bf` | `int` |

**Returns:** `int`


#### `extern fn timeout_wrapper(_delay: int) -> void`

**Parameters:**
| Name | Type |
|------|------|
| `_delay` | `int` |

**Returns:** `void`


#### `extern fn start_color_wrapper() -> int`

**Returns:** `int`


#### `extern fn has_colors_wrapper() -> int`

**Returns:** `int`


#### `extern fn init_pair_wrapper(_pair: int, _fg: int, _bg: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_pair` | `int` |
| `_fg` | `int` |
| `_bg` | `int` |

**Returns:** `int`


#### `extern fn attron_wrapper(_attrs: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_attrs` | `int` |

**Returns:** `int`


#### `extern fn attroff_wrapper(_attrs: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_attrs` | `int` |

**Returns:** `int`


#### `extern fn getmaxx_wrapper(_win: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_win` | `int` |

**Returns:** `int`


#### `extern fn getmaxy_wrapper(_win: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_win` | `int` |

**Returns:** `int`


#### `extern fn stdscr_wrapper() -> int`

**Returns:** `int`


#### `extern fn noecho_wrapper() -> int`

**Returns:** `int`


#### `extern fn echo_wrapper() -> int`

**Returns:** `int`


#### `extern fn cbreak_wrapper() -> int`

**Returns:** `int`


#### `extern fn nocbreak_wrapper() -> int`

**Returns:** `int`


#### `extern fn box_wrapper(_win: int, _verch: int, _horch: int) -> int`

**Parameters:**
| Name | Type |
|------|------|
| `_win` | `int` |
| `_verch` | `int` |
| `_horch` | `int` |

**Returns:** `int`


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
| `COLOR_BLACK` | `int` | `0` |
| `COLOR_RED` | `int` | `1` |
| `COLOR_GREEN` | `int` | `2` |
| `COLOR_YELLOW` | `int` | `3` |
| `COLOR_BLUE` | `int` | `4` |
| `COLOR_MAGENTA` | `int` | `5` |
| `COLOR_CYAN` | `int` | `6` |
| `COLOR_WHITE` | `int` | `7` |
| `KEY_DOWN` | `int` | `258` |
| `KEY_UP` | `int` | `259` |
| `KEY_LEFT` | `int` | `260` |
| `KEY_RIGHT` | `int` | `261` |
| `KEY_HOME` | `int` | `262` |
| `KEY_BACKSPACE` | `int` | `263` |
| `KEY_F0` | `int` | `264` |
| `KEY_ESC` | `int` | `27` |
| `KEY_SPACE` | `int` | `32` |
| `KEY_ENTER` | `int` | `10` |
| `KEY_DC` | `int` | `330` |
| `KEY_IC` | `int` | `331` |
| `KEY_PPAGE` | `int` | `339` |
| `KEY_NPAGE` | `int` | `338` |
| `KEY_END` | `int` | `360` |
| `ERR` | `int` | `-1` |
| `OK` | `int` | `0` |
| `A_NORMAL` | `int` | `0` |
| `A_STANDOUT` | `int` | `65536` |
| `A_UNDERLINE` | `int` | `131072` |
| `A_REVERSE` | `int` | `262144` |
| `A_BLINK` | `int` | `524288` |
| `A_DIM` | `int` | `1048576` |
| `A_BOLD` | `int` | `2097152` |

