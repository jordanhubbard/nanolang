#include <ncurses.h>
#include <stdint.h>
#include <stdlib.h>

// Wrapper for initscr - returns pointer as int64_t
int64_t initscr_wrapper() {
    WINDOW* win = initscr();
    return (int64_t)win;
}

// Wrapper for endwin
int64_t endwin_wrapper() {
    return (int64_t)endwin();
}

// Wrapper for curs_set
int64_t curs_set_wrapper(int64_t visibility) {
    return (int64_t)curs_set((int)visibility);
}

// Wrapper for clear
int64_t clear_wrapper() {
    return (int64_t)clear();
}

// Wrapper for refresh
int64_t refresh_wrapper() {
    return (int64_t)refresh();
}

// Wrapper for mvprintw
int64_t mvprintw_wrapper(int64_t y, int64_t x, const char* str) {
    return (int64_t)mvprintw((int)y, (int)x, "%s", str);
}

// Wrapper for mvaddch
int64_t mvaddch_wrapper(int64_t y, int64_t x, int64_t ch) {
    return (int64_t)mvaddch((int)y, (int)x, (int)ch);
}

// Wrapper for getch
int64_t getch_wrapper() {
    return (int64_t)getch();
}

// Wrapper for nodelay
int64_t nl_nodelay(int64_t win, int64_t bf) {
    WINDOW* w = (win == 0) ? stdscr : (WINDOW*)win;
    return (int64_t)nodelay(w, (int)bf);
}

// Wrapper for keypad
int64_t nl_keypad(int64_t win, int64_t bf) {
    WINDOW* w = (win == 0) ? stdscr : (WINDOW*)win;
    return (int64_t)keypad(w, (int)bf);
}

// Wrapper for timeout
void timeout_wrapper(int64_t delay) {
    timeout((int)delay);
}

// Wrapper for addch
int64_t addch_wrapper(int64_t ch) {
    return (int64_t)addch((int)ch);
}

// Wrapper for addstr
int64_t addstr_wrapper(const char* str) {
    return (int64_t)addstr(str);
}

// Wrapper for move
int64_t move_wrapper(int64_t y, int64_t x) {
    return (int64_t)move((int)y, (int)x);
}

// Wrapper for erase
int64_t erase_wrapper() {
    return (int64_t)erase();
}

// Wrapper for start_color
int64_t start_color_wrapper() {
    return (int64_t)start_color();
}

// Wrapper for has_colors
int64_t has_colors_wrapper() {
    return (int64_t)has_colors();
}

// Wrapper for init_pair
int64_t init_pair_wrapper(int64_t pair, int64_t fg, int64_t bg) {
    return (int64_t)init_pair((int)pair, (int)fg, (int)bg);
}

// Wrapper for attron
int64_t attron_wrapper(int64_t attrs) {
    return (int64_t)attron((int)attrs);
}

// Wrapper for attroff
int64_t attroff_wrapper(int64_t attrs) {
    return (int64_t)attroff((int)attrs);
}

// Wrapper for getmaxx
int64_t getmaxx_wrapper(int64_t win) {
    WINDOW* w = (win == 0) ? stdscr : (WINDOW*)win;
    return (int64_t)getmaxx(w);
}

// Wrapper for getmaxy
int64_t getmaxy_wrapper(int64_t win) {
    WINDOW* w = (win == 0) ? stdscr : (WINDOW*)win;
    return (int64_t)getmaxy(w);
}

// Wrapper for stdscr
int64_t stdscr_wrapper() {
    return (int64_t)stdscr;
}

// Additional useful functions

// Wrapper for noecho
int64_t noecho_wrapper() {
    return (int64_t)noecho();
}

// Wrapper for echo
int64_t echo_wrapper() {
    return (int64_t)echo();
}

// Wrapper for cbreak
int64_t cbreak_wrapper() {
    return (int64_t)cbreak();
}

// Wrapper for nocbreak
int64_t nocbreak_wrapper() {
    return (int64_t)nocbreak();
}

// Wrapper for box
int64_t box_wrapper(int64_t win, int64_t verch, int64_t horch) {
    WINDOW* w = (win == 0) ? stdscr : (WINDOW*)win;
    return (int64_t)box(w, (chtype)verch, (chtype)horch);
}

// Wrapper for mvaddstr
int64_t mvaddstr_wrapper(int64_t y, int64_t x, const char* str) {
    return (int64_t)mvaddstr((int)y, (int)x, str);
}
