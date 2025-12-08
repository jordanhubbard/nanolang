# NCurses Module Review

## Current Status: FUNCTIONAL BUT INCOMPLETE

The module works (snake_ncurses compiles and should run), but has several issues and missing features.

---

## 🔴 CRITICAL ISSUES

### 1. **Inconsistent Naming Convention**
**Problem:** Mix of `_wrapper` suffix and no suffix
```nano
# Has wrapper suffix:
extern fn initscr_wrapper() -> int
extern fn mvprintw_wrapper(y: int, x: int, str: string) -> int

# No wrapper suffix:
extern fn addch(ch: int) -> int
extern fn move(y: int, x: int) -> int
extern fn erase() -> int
```

**Impact:** Confusing API - users never know if they need `_wrapper` or not

**Fix:** Choose ONE approach:
- **Option A:** All functions have `_wrapper` suffix (consistent but ugly)
- **Option B:** Create wrapper functions WITHOUT suffix that call C functions (better UX)
- **Option C:** No wrappers, just extern declarations (simplest but no type conversions)

**Recommendation:** Option B - Define all C functions as `_wrapper`, then create nano wrapper functions:
```nano
extern fn mvprintw_wrapper(y: int, x: int, str: string) -> int
fn mvprintw(y: int, x: int, str: string) -> int {
    return (mvprintw_wrapper y x str)
}
```

---

### 2. **Missing C Wrappers for Some Functions**
**Problem:** Some functions declared but not implemented in C:
```nano
# Declared in .nano but NO C wrapper:
extern fn addch(ch: int) -> int              # ❌ Not in C file
extern fn addstr(str: string) -> int         # ❌ Not in C file
extern fn move(y: int, x: int) -> int        # ❌ Not in C file
extern fn erase() -> int                     # ❌ Not in C file
extern fn start_color() -> int               # ❌ Not in C file
extern fn has_colors() -> int                # ❌ Not in C file
extern fn init_pair(...) -> int              # ❌ Not in C file
extern fn attron(attrs: int) -> int          # ❌ Not in C file
extern fn attroff(attrs: int) -> int         # ❌ Not in C file
extern fn getmaxx(win: int) -> int           # ❌ Not in C file
extern fn getmaxy(win: int) -> int           # ❌ Not in C file
extern fn stdscr() -> int                    # ❌ Not in C file
```

**Impact:** Linker errors when trying to use these functions

**Fix:** Add wrappers for all declared functions OR remove declarations

---

### 3. **Color Support Missing Wrappers**
**Problem:** COLOR_PAIR macro not available
```nano
# Need this to use color pairs:
fn COLOR_PAIR(n: int) -> int {
    return (<< n 8)  # shift left by 8 bits
}
```

**Impact:** Can't actually use colors even though constants are defined

---

## ⚠️ MISSING IMPORTANT NCURSES FUNCTIONS

### **Window Management** (for panels/multiple windows)
```c
WINDOW* newwin(int nlines, int ncols, int begin_y, int begin_x);
void delwin(WINDOW* win);
void wrefresh(WINDOW* win);
void wclear(WINDOW* win);
```

### **Box Drawing**
```c
int box(WINDOW* win, chtype verch, chtype horch);  // Draw box around window
```

### **Attribute Management**
```c
int attrset(int attrs);           // Set attributes
int standout();                   // Start standout mode
int standend();                   // End standout mode
```

### **Text Output**
```c
int printw(const char* fmt, ...);           // Print to cursor position
int mvaddstr(int y, int x, const char* str); // Move and add string
int addnstr(const char* str, int n);        // Add n characters
```

### **Cursor Control**
```c
int getyx(WINDOW* win, int& y, int& x);  // Get cursor position
```

### **Input Control**
```c
int noecho();              // Don't echo typed characters
int echo();                // Echo typed characters
int cbreak();              // Disable line buffering
int nocbreak();            // Enable line buffering
int raw();                 // Raw input mode
int noraw();               // No raw input
int halfdelay(int tenths); // Half-delay mode
```

### **Screen Updates**
```c
int doupdate();            // Update physical screen
int wnoutrefresh(WINDOW*); // Mark for refresh but don't update
```

### **Special Characters**
```c
// ACS_* constants for box drawing:
ACS_ULCORNER, ACS_URCORNER, ACS_LLCORNER, ACS_LRCORNER
ACS_HLINE, ACS_VLINE
ACS_PLUS, ACS_LTEE, ACS_RTEE, ACS_BTEE, ACS_TTEE
```

---

## 🟡 DESIGN ISSUES

### 1. **Window Handle as int**
**Current:**
```nano
extern fn initscr_wrapper() -> int
```

**Problem:** Using `int` to represent WINDOW* pointer is fragile
- On 64-bit systems, pointers are 64-bit but nanolang int is 64-bit (OK)
- But it's semantically wrong - it's not really an int
- Can't do type checking

**Better approach (if nanolang supported it):**
```nano
opaque type WINDOW
extern fn initscr_wrapper() -> WINDOW
```
But this was causing issues, so using `int` is a pragmatic workaround.

**Status:** ✅ Acceptable given nanolang limitations

---

### 2. **Boolean Parameters as int**
**Current:**
```nano
extern fn nl_nodelay(win: int, bf: int) -> int
```

**Problem:** `bf` should be boolean (true/false) but we use int (0/1)

**Better:**
```nano
extern fn nl_nodelay(win: int, bf: bool) -> int
# Then in C:
int64_t nl_nodelay(int64_t win, bool bf) {
    return nodelay((WINDOW*)win, bf);
}
```

**Status:** Minor issue - current approach works but less type-safe

---

### 3. **Error Handling Missing**
**Problem:** No way to check for errors
- Most ncurses functions return ERR (-1) on error
- We return the error code but users can't compare to ERR

**Fix:** Add error constant:
```nano
let ERR: int = -1
let OK: int = 0
```

---

## 📋 RECOMMENDED IMPROVEMENTS

### **Priority 1: Fix Critical Issues**

1. **Create C wrappers for all declared functions:**
```c
// In ncurses_helpers.c:
int64_t addch_wrapper(int64_t ch) {
    return (int64_t)addch((int)ch);
}

int64_t addstr_wrapper(const char* str) {
    return (int64_t)addstr(str);
}

int64_t move_wrapper(int64_t y, int64_t x) {
    return (int64_t)move((int)y, (int)x);
}

int64_t erase_wrapper() {
    return (int64_t)erase();
}

// Color functions:
int64_t start_color_wrapper() {
    return (int64_t)start_color();
}

int64_t has_colors_wrapper() {
    return (int64_t)has_colors();
}

int64_t init_pair_wrapper(int64_t pair, int64_t fg, int64_t bg) {
    return (int64_t)init_pair((int)pair, (int)fg, (int)bg);
}

int64_t attron_wrapper(int64_t attrs) {
    return (int64_t)attron((int)attrs);
}

int64_t attroff_wrapper(int64_t attrs) {
    return (int64_t)attroff((int)attrs);
}

// Screen size functions:
int64_t getmaxx_wrapper(int64_t win) {
    WINDOW* w = (win == 0) ? stdscr : (WINDOW*)win;
    return (int64_t)getmaxx(w);
}

int64_t getmaxy_wrapper(int64_t win) {
    WINDOW* w = (win == 0) ? stdscr : (WINDOW*)win;
    return (int64_t)getmaxy(w);
}

int64_t stdscr_wrapper() {
    return (int64_t)stdscr;
}
```

2. **Update ncurses.nano with _wrapper suffix:**
```nano
extern fn addch_wrapper(ch: int) -> int
extern fn addstr_wrapper(str: string) -> int
extern fn move_wrapper(y: int, x: int) -> int
extern fn erase_wrapper() -> int
extern fn start_color_wrapper() -> int
extern fn has_colors_wrapper() -> int
extern fn init_pair_wrapper(pair: int, fg: int, bg: int) -> int
extern fn attron_wrapper(attrs: int) -> int
extern fn attroff_wrapper(attrs: int) -> int
extern fn getmaxx_wrapper(win: int) -> int
extern fn getmaxy_wrapper(win: int) -> int
extern fn stdscr_wrapper() -> int
```

3. **Create convenience wrappers (no suffix):**
```nano
# === CONVENIENCE FUNCTIONS ===
# User-friendly names without _wrapper suffix

fn initscr() -> int { return (initscr_wrapper) }
fn endwin() -> int { return (endwin_wrapper) }
fn curs_set(visibility: int) -> int { return (curs_set_wrapper visibility) }
fn clear() -> int { return (clear_wrapper) }
fn refresh() -> int { return (refresh_wrapper) }
fn mvprintw(y: int, x: int, str: string) -> int { 
    return (mvprintw_wrapper y x str) 
}
fn mvaddch(y: int, x: int, ch: int) -> int { 
    return (mvaddch_wrapper y x ch) 
}
fn getch() -> int { return (getch_wrapper) }
fn timeout(delay: int) -> void { (timeout_wrapper delay) }
fn addch(ch: int) -> int { return (addch_wrapper ch) }
fn addstr(str: string) -> int { return (addstr_wrapper str) }
fn move(y: int, x: int) -> int { return (move_wrapper y x) }
fn erase() -> int { return (erase_wrapper) }
fn start_color() -> int { return (start_color_wrapper) }
fn has_colors() -> int { return (has_colors_wrapper) }
fn init_pair(pair: int, fg: int, bg: int) -> int {
    return (init_pair_wrapper pair fg bg)
}
fn attron(attrs: int) -> int { return (attron_wrapper attrs) }
fn attroff(attrs: int) -> int { return (attroff_wrapper attrs) }
fn getmaxx(win: int) -> int { return (getmaxx_wrapper win) }
fn getmaxy(win: int) -> int { return (getmaxy_wrapper win) }
fn stdscr() -> int { return (stdscr_wrapper) }
fn nodelay(win: int, bf: int) -> int { return (nl_nodelay win bf) }
fn keypad(win: int, bf: int) -> int { return (nl_keypad win bf) }
```

---

### **Priority 2: Add Essential Missing Functions**

```c
// Input control:
int64_t noecho_wrapper() { return (int64_t)noecho(); }
int64_t echo_wrapper() { return (int64_t)echo(); }
int64_t cbreak_wrapper() { return (int64_t)cbreak(); }
int64_t nocbreak_wrapper() { return (int64_t)nocbreak(); }

// Box drawing:
int64_t box_wrapper(int64_t win, int64_t verch, int64_t horch) {
    WINDOW* w = (win == 0) ? stdscr : (WINDOW*)win;
    return (int64_t)box(w, (chtype)verch, (chtype)horch);
}

// More text output:
int64_t mvaddstr_wrapper(int64_t y, int64_t x, const char* str) {
    return (int64_t)mvaddstr((int)y, (int)x, str);
}

int64_t printw_wrapper(const char* str) {
    return (int64_t)printw("%s", str);
}
```

---

### **Priority 3: Add Useful Constants**

```nano
# Error codes
let ERR: int = -1
let OK: int = 0

# Attributes
let A_NORMAL: int = 0
let A_STANDOUT: int = 65536
let A_UNDERLINE: int = 131072
let A_REVERSE: int = 262144
let A_BLINK: int = 524288
let A_DIM: int = 1048576
let A_BOLD: int = 2097152

# More special keys
let KEY_DC: int = 330        # Delete character
let KEY_IC: int = 331        # Insert character
let KEY_PPAGE: int = 339     # Page up
let KEY_NPAGE: int = 338     # Page down
let KEY_END: int = 360       # End key
let KEY_F1: int = 265
let KEY_F2: int = 266
# ... F3-F12

# Box drawing characters (ACS_*)
let ACS_ULCORNER: int = 4194412
let ACS_URCORNER: int = 4194411
let ACS_LLCORNER: int = 4194410
let ACS_LRCORNER: int = 4194409
let ACS_HLINE: int = 4194417
let ACS_VLINE: int = 4194424
```

---

### **Priority 4: Add COLOR_PAIR Helper**

```nano
# Helper to create color pair attribute
fn COLOR_PAIR(n: int) -> int {
    return (<< n 8)
}

# Usage example:
# (init_pair 1 COLOR_RED COLOR_BLACK)
# (attron (COLOR_PAIR 1))
# (mvprintw 5 5 "Red text!")
# (attroff (COLOR_PAIR 1))
```

---

## 📝 SUMMARY

### **What Works:**
✅ Basic screen initialization/cleanup
✅ Text output (mvprintw, mvaddch)
✅ Input (getch with timeout)
✅ Non-blocking input setup
✅ Module builds and links correctly

### **What's Broken:**
❌ Many declared functions have no C implementation (linker errors if used)
❌ Color functions declared but not usable (no wrappers + no COLOR_PAIR)
❌ Inconsistent naming (_wrapper vs no wrapper)

### **What's Missing:**
- Input mode control (echo, cbreak, raw)
- Box drawing functions
- Attribute management
- Screen size queries (getmaxx/getmaxy need wrappers)
- Error constants
- More key codes
- Box drawing character constants

---

## 🎯 RECOMMENDATION

**Fix in this order:**

1. **Immediate (blocking):** Add C wrappers for all declared functions
2. **High priority:** Make naming consistent (all _wrapper, then convenience functions)
3. **High priority:** Add COLOR_PAIR helper and fix color support
4. **Medium priority:** Add echo/cbreak/box functions
5. **Low priority:** Add more key codes and ACS constants

**Minimal fix to make current code work:**
- Add the 12 missing C wrappers for functions already declared
- Update ncurses_helpers.h with declarations
- Then both snake and game_of_life will work

**Result:** Fully functional basic ncurses module suitable for terminal games and UIs!
