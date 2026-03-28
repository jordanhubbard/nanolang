# 19.1 ncurses — Terminal Applications

**Create interactive text-based user interfaces in the terminal.**

The `ncurses` module exposes the ncurses library through a thin FFI layer. It lets you take full control of the terminal: move the cursor to any position, draw text with colors and attributes, read single keystrokes without waiting for Enter, and query the terminal size.

All functions use a `_wrapper` suffix on the C side, but you import them directly using their natural names. The module also exports color constants (`COLOR_RED`, etc.), attribute constants (`A_BOLD`, etc.), and key code constants (`KEY_UP`, `KEY_ESC`, etc.).

## Quick Start

```nano
from "modules/ncurses/ncurses.nano" import initscr_wrapper, endwin_wrapper,
                                            noecho_wrapper, cbreak_wrapper,
                                            getch_wrapper, refresh_wrapper,
                                            mvprintw_wrapper, clear_wrapper

fn hello_terminal() -> void {
    (initscr_wrapper)    # Enter curses mode
    (noecho_wrapper)     # Don't echo typed characters
    (cbreak_wrapper)     # Disable line buffering

    (mvprintw_wrapper 0 0 "Hello, Terminal!")
    (mvprintw_wrapper 1 0 "Press any key to exit...")
    (refresh_wrapper)

    (getch_wrapper)      # Wait for one keystroke

    (endwin_wrapper)     # Restore normal terminal
}

shadow hello_terminal {
    # Cannot run interactive TUI in shadow tests; just assert compile success
    assert true
}
```

## Initialisation and Teardown

Every ncurses program follows the same lifecycle:

```nano
# 1. Enter curses mode
(initscr_wrapper)

# ... set up input modes, colors, draw ...

# 2. Run your UI loop

# 3. Return to normal terminal
(endwin_wrapper)
```

Forgetting `endwin_wrapper` leaves the terminal in curses mode, which garbles subsequent shell output.

### Input Mode Setup

After `initscr_wrapper`, configure how the terminal handles input:

| Call | Effect |
|---|---|
| `(noecho_wrapper)` | Typed characters are not printed to screen |
| `(echo_wrapper)` | Characters are echoed (default) |
| `(cbreak_wrapper)` | Input available immediately, no line buffering |
| `(nocbreak_wrapper)` | Restore default line-buffered input |
| `(curs_set_wrapper 0)` | Hide the cursor |
| `(curs_set_wrapper 1)` | Show normal cursor (default) |
| `(curs_set_wrapper 2)` | Show high-visibility cursor |

For interactive games and forms you almost always want `noecho` + `cbreak`:

```nano
from "modules/ncurses/ncurses.nano" import initscr_wrapper, noecho_wrapper,
                                            cbreak_wrapper, curs_set_wrapper

fn setup_tui() -> void {
    (initscr_wrapper)
    (noecho_wrapper)
    (cbreak_wrapper)
    (curs_set_wrapper 0)   # Hide cursor for cleaner look
}

shadow setup_tui {
    assert true
}
```

## Moving and Printing Text

### Moving the Cursor

```
move_wrapper(y: int, x: int) -> int
```

Move the cursor to row `y`, column `x`. The top-left corner is `(0, 0)`. Must call `refresh_wrapper` for changes to appear.

```nano
from "modules/ncurses/ncurses.nano" import move_wrapper, addstr_wrapper, refresh_wrapper

fn draw_status_bar(row: int, message: string) -> void {
    (move_wrapper row 0)
    (addstr_wrapper message)
    (refresh_wrapper)
}

shadow draw_status_bar {
    assert true
}
```

### Move-and-Print (mvprintw)

The most convenient output function prints a string at a specific location in one call:

```
mvprintw_wrapper(y: int, x: int, text: string) -> int
```

```nano
from "modules/ncurses/ncurses.nano" import mvprintw_wrapper, refresh_wrapper

fn draw_score(score: int) -> void {
    let text: string = (+ "Score: " (int_to_string score))
    (mvprintw_wrapper 0 0 text)
    (refresh_wrapper)
}

shadow draw_score {
    assert true
}
```

### Other Output Functions

| Function | Description |
|---|---|
| `addstr_wrapper(str)` | Print string at current cursor position |
| `addch_wrapper(ch)` | Print one character (as int ASCII code) |
| `mvaddstr_wrapper(y, x, str)` | Move then print string |
| `mvaddch_wrapper(y, x, ch)` | Move then print character |
| `clear_wrapper()` | Clear entire screen |
| `erase_wrapper()` | Erase entire screen (softer than clear) |
| `refresh_wrapper()` | Flush pending changes to the real terminal |

### Getting Terminal Dimensions

```
stdscr_wrapper() -> int
getmaxx_wrapper(win: int) -> int
getmaxy_wrapper(win: int) -> int
```

```nano
from "modules/ncurses/ncurses.nano" import initscr_wrapper, stdscr_wrapper,
                                            getmaxx_wrapper, getmaxy_wrapper,
                                            endwin_wrapper

fn get_terminal_size() -> void {
    (initscr_wrapper)
    let scr: int = (stdscr_wrapper)
    let cols: int = (getmaxx_wrapper scr)
    let rows: int = (getmaxy_wrapper scr)
    (endwin_wrapper)
    (println (+ "Terminal: " (+ (int_to_string cols) (+ "x" (int_to_string rows)))))
}

shadow get_terminal_size {
    assert true
}
```

## Keyboard Input

### Blocking Input

```
getch_wrapper() -> int
```

Wait for a keypress and return its key code as an `int`. For printable ASCII characters the code equals the ASCII value. Special keys use the `KEY_*` constants.

```nano
from "modules/ncurses/ncurses.nano" import getch_wrapper, KEY_UP, KEY_DOWN,
                                            KEY_LEFT, KEY_RIGHT, KEY_ESC

fn handle_key(ch: int) -> string {
    if (== ch KEY_UP) {
        return "up"
    } else {
        if (== ch KEY_DOWN) {
            return "down"
        } else {
            if (== ch KEY_LEFT) {
                return "left"
            } else {
                if (== ch KEY_RIGHT) {
                    return "right"
                } else {
                    if (== ch KEY_ESC) {
                        return "quit"
                    } else {
                        return "other"
                    }
                }
            }
        }
    }
}

shadow handle_key {
    assert (== (handle_key KEY_UP) "up")
    assert (== (handle_key KEY_ESC) "quit")
    assert (== (handle_key 65) "other")   # 'A'
}
```

### Non-Blocking Input

For game loops you often want `getch` to return immediately with `ERR` (`-1`) if no key is pressed, rather than blocking:

```
timeout_wrapper(delay_ms: int) -> void
nl_nodelay(win: int, flag: int) -> int
```

Use `timeout_wrapper` with a negative delay to block, `0` for non-blocking, or a positive value for a millisecond timeout:

```nano
from "modules/ncurses/ncurses.nano" import timeout_wrapper, getch_wrapper, ERR

fn poll_key() -> int {
    (timeout_wrapper 0)   # non-blocking
    return (getch_wrapper)
}

shadow poll_key {
    assert true
}
```

### Enabling Arrow Keys

By default, function and arrow keys may not be decoded. Enable keypad mode on the standard screen:

```nano
from "modules/ncurses/ncurses.nano" import initscr_wrapper, stdscr_wrapper, nl_keypad

fn enable_arrow_keys() -> void {
    (initscr_wrapper)
    let scr: int = (stdscr_wrapper)
    (nl_keypad scr 1)   # 1 = enable, 0 = disable
}

shadow enable_arrow_keys {
    assert true
}
```

## Colors

ncurses uses a **color pair** system. You define pairs of (foreground, background) colors and then activate a pair when drawing.

### Setup

```nano
from "modules/ncurses/ncurses.nano" import initscr_wrapper, start_color_wrapper,
                                            has_colors_wrapper, init_pair_wrapper,
                                            COLOR_RED, COLOR_BLACK, COLOR_GREEN,
                                            COLOR_WHITE, COLOR_BLUE, COLOR_YELLOW

fn setup_colors() -> void {
    (initscr_wrapper)
    if (== (has_colors_wrapper) 1) {
        (start_color_wrapper)
        # Define color pairs: (pair_number, foreground, background)
        (init_pair_wrapper 1 COLOR_RED   COLOR_BLACK)
        (init_pair_wrapper 2 COLOR_GREEN COLOR_BLACK)
        (init_pair_wrapper 3 COLOR_WHITE COLOR_BLUE)
        (init_pair_wrapper 4 COLOR_BLACK COLOR_YELLOW)
    } else {
        (print "")
    }
}

shadow setup_colors {
    assert true
}
```

### Drawing with Color

Use `attron_wrapper` and `COLOR_PAIR` to activate a color pair, draw, then `attroff_wrapper` to deactivate:

```nano
from "modules/ncurses/ncurses.nano" import attron_wrapper, attroff_wrapper,
                                            mvprintw_wrapper, refresh_wrapper,
                                            COLOR_PAIR

fn draw_colored(y: int, x: int, pair: int, text: string) -> void {
    let attr: int = (COLOR_PAIR pair)
    (attron_wrapper attr)
    (mvprintw_wrapper y x text)
    (attroff_wrapper attr)
    (refresh_wrapper)
}

shadow draw_colored {
    assert true
}
```

### Available Color Constants

| Constant | Value | Color |
|---|---|---|
| `COLOR_BLACK` | 0 | Black |
| `COLOR_RED` | 1 | Red |
| `COLOR_GREEN` | 2 | Green |
| `COLOR_YELLOW` | 3 | Yellow |
| `COLOR_BLUE` | 4 | Blue |
| `COLOR_MAGENTA` | 5 | Magenta |
| `COLOR_CYAN` | 6 | Cyan |
| `COLOR_WHITE` | 7 | White |

### Text Attributes

Combine with `attron_wrapper` / `attroff_wrapper`:

| Constant | Effect |
|---|---|
| `A_NORMAL` | Normal text |
| `A_BOLD` | Bold / bright |
| `A_DIM` | Dimmed |
| `A_UNDERLINE` | Underlined |
| `A_REVERSE` | Reverse video (swap fg/bg) |
| `A_BLINK` | Blinking (terminal-dependent) |
| `A_STANDOUT` | Terminal's "best" highlighting mode |

```nano
from "modules/ncurses/ncurses.nano" import attron_wrapper, attroff_wrapper,
                                            mvprintw_wrapper, refresh_wrapper,
                                            A_BOLD, A_UNDERLINE

fn draw_title(text: string) -> void {
    (attron_wrapper A_BOLD)
    (attron_wrapper A_UNDERLINE)
    (mvprintw_wrapper 0 0 text)
    (attroff_wrapper A_UNDERLINE)
    (attroff_wrapper A_BOLD)
    (refresh_wrapper)
}

shadow draw_title {
    assert true
}
```

## Drawing Boxes

Use `box_wrapper` to draw a border around a window. Pass `0` for the default border characters:

```nano
from "modules/ncurses/ncurses.nano" import initscr_wrapper, stdscr_wrapper,
                                            box_wrapper, refresh_wrapper, endwin_wrapper,
                                            getch_wrapper

fn draw_border() -> void {
    (initscr_wrapper)
    let scr: int = (stdscr_wrapper)
    (box_wrapper scr 0 0)
    (refresh_wrapper)
    (getch_wrapper)
    (endwin_wrapper)
}

shadow draw_border {
    assert true
}
```

## Complete Example: Interactive Menu

A minimal interactive menu demonstrating the full lifecycle:

```nano
from "modules/ncurses/ncurses.nano" import initscr_wrapper, endwin_wrapper,
                                            noecho_wrapper, cbreak_wrapper,
                                            curs_set_wrapper, stdscr_wrapper,
                                            nl_keypad, start_color_wrapper,
                                            has_colors_wrapper, init_pair_wrapper,
                                            attron_wrapper, attroff_wrapper,
                                            mvprintw_wrapper, clear_wrapper,
                                            refresh_wrapper, getch_wrapper,
                                            box_wrapper, getmaxy_wrapper,
                                            getmaxx_wrapper,
                                            KEY_UP, KEY_DOWN, KEY_ENTER, KEY_ESC,
                                            COLOR_PAIR, COLOR_WHITE, COLOR_BLUE,
                                            COLOR_BLACK, A_BOLD

let mut selected_item: int = 0
let MENU_ITEMS: int = 3

fn draw_menu() -> void {
    (clear_wrapper)

    let scr: int = (stdscr_wrapper)
    let rows: int = (getmaxy_wrapper scr)
    let cols: int = (getmaxx_wrapper scr)

    # Title
    (attron_wrapper A_BOLD)
    (mvprintw_wrapper 1 2 "Simple Menu Demo")
    (attroff_wrapper A_BOLD)

    # Menu items
    let mut i: int = 0
    while (< i MENU_ITEMS) {
        let label: string = (cond
            ((== i 0) "  New Game  ")
            ((== i 1) "  Options   ")
            (else      "  Quit      ")
        )
        let row: int = (+ 3 i)
        if (== i selected_item) {
            (attron_wrapper (COLOR_PAIR 1))
            (mvprintw_wrapper row 2 label)
            (attroff_wrapper (COLOR_PAIR 1))
        } else {
            (mvprintw_wrapper row 2 label)
        }
        set i (+ i 1)
    }

    (mvprintw_wrapper (- rows 2) 2 "Use arrows to navigate, Enter to select, Esc to quit")
    (refresh_wrapper)
}

shadow draw_menu {
    set selected_item 0
    assert true
}

fn run_menu() -> int {
    (initscr_wrapper)
    (noecho_wrapper)
    (cbreak_wrapper)
    (curs_set_wrapper 0)

    let scr: int = (stdscr_wrapper)
    (nl_keypad scr 1)

    if (== (has_colors_wrapper) 1) {
        (start_color_wrapper)
        (init_pair_wrapper 1 COLOR_WHITE COLOR_BLUE)
    } else {
        (print "")
    }

    set selected_item 0
    let mut running: bool = true
    while running {
        (draw_menu)
        let ch: int = (getch_wrapper)
        if (== ch KEY_UP) {
            if (> selected_item 0) {
                set selected_item (- selected_item 1)
            } else {
                (print "")
            }
        } else {
            if (== ch KEY_DOWN) {
                if (< selected_item (- MENU_ITEMS 1)) {
                    set selected_item (+ selected_item 1)
                } else {
                    (print "")
                }
            } else {
                if (== ch KEY_ENTER) {
                    set running false
                } else {
                    if (== ch KEY_ESC) {
                        set selected_item (- MENU_ITEMS 1)   # Quit
                        set running false
                    } else {
                        (print "")
                    }
                }
            }
        }
    }

    (endwin_wrapper)
    return selected_item
}

shadow run_menu {
    assert true
}
```

## Key Code Reference

| Constant | Value | Key |
|---|---|---|
| `KEY_DOWN` | 258 | Down arrow |
| `KEY_UP` | 259 | Up arrow |
| `KEY_LEFT` | 260 | Left arrow |
| `KEY_RIGHT` | 261 | Right arrow |
| `KEY_HOME` | 262 | Home |
| `KEY_BACKSPACE` | 263 | Backspace |
| `KEY_ESC` | 27 | Escape |
| `KEY_SPACE` | 32 | Space |
| `KEY_ENTER` | 10 | Enter / Return |
| `KEY_DC` | 330 | Delete |
| `KEY_PPAGE` | 339 | Page Up |
| `KEY_NPAGE` | 338 | Page Down |
| `KEY_END` | 360 | End |
| `ERR` | -1 | No key (non-blocking) |
| `OK` | 0 | Success return code |

---

**Previous:** [Chapter 19 Overview](index.html)
**Next:** [Chapter 20: Testing](../20_testing/index.html)
