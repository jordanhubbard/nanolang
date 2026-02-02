# Chapter 19: Terminal UI (ncurses)

**Text-based user interfaces in the terminal.**

ncurses provides terminal control for creating TUIs (Text User Interfaces).

## 19.1 Initialization

```nano
from "modules/ncurses/ncurses.nano" import initscr, endwin, getch

fn run_tui() -> int {
    (initscr)
    let ch: int = (getch)
    (endwin)
    return ch
}

shadow run_tui {
    assert true
}
```

## 19.2 Text Output

```nano
from "modules/ncurses/ncurses.nano" import printw, move, refresh

fn display_text() -> void {
    (move 5 10)
    (printw "Hello, Terminal!")
    (refresh)
}

shadow display_text {
    assert true
}
```

## 19.3 Colors

```nano
from "modules/ncurses/ncurses.nano" import start_color, init_pair, attron, attroff

fn colored_text() -> void {
    (start_color)
    (init_pair 1 "red" "black")
    (attron 1)
    (printw "Red text")
    (attroff 1)
}

shadow colored_text {
    assert true
}
```

## 19.4 Windows

```nano
from "modules/ncurses/ncurses.nano" import newwin, wrefresh, delwin

fn create_subwindow() -> void {
    let win: Window = (newwin 10 40 5 5)
    (wrefresh win)
    (delwin win)
}

shadow create_subwindow {
    assert true
}
```

## Summary

ncurses provides:
- ✅ Terminal control
- ✅ Colored text
- ✅ Windows and panels
- ✅ Keyboard input

---

**Previous:** [Chapter 18: Game Development](../18_game_dev/index.md)  
**Next:** [Chapter 20: Testing & Quality](../20_testing/index.md)
