#ifndef NCURSES_HELPERS_H
#define NCURSES_HELPERS_H

#include <stdint.h>

// Core wrappers for ncurses functions
int64_t initscr_wrapper();
int64_t endwin_wrapper();
int64_t curs_set_wrapper(int64_t visibility);
int64_t clear_wrapper();
int64_t refresh_wrapper();
int64_t mvprintw_wrapper(int64_t y, int64_t x, const char* str);
int64_t mvaddch_wrapper(int64_t y, int64_t x, int64_t ch);
int64_t getch_wrapper();
int64_t nl_nodelay(int64_t win, int64_t bf);
int64_t nl_keypad(int64_t win, int64_t bf);
void timeout_wrapper(int64_t delay);

// Text output wrappers
int64_t addch_wrapper(int64_t ch);
int64_t addstr_wrapper(const char* str);
int64_t mvaddstr_wrapper(int64_t y, int64_t x, const char* str);
int64_t move_wrapper(int64_t y, int64_t x);
int64_t erase_wrapper();

// Color wrappers
int64_t start_color_wrapper();
int64_t has_colors_wrapper();
int64_t init_pair_wrapper(int64_t pair, int64_t fg, int64_t bg);
int64_t attron_wrapper(int64_t attrs);
int64_t attroff_wrapper(int64_t attrs);

// Screen info wrappers
int64_t getmaxx_wrapper(int64_t win);
int64_t getmaxy_wrapper(int64_t win);
int64_t stdscr_wrapper();

// Input mode wrappers
int64_t noecho_wrapper();
int64_t echo_wrapper();
int64_t cbreak_wrapper();
int64_t nocbreak_wrapper();

// Box drawing wrapper
int64_t box_wrapper(int64_t win, int64_t verch, int64_t horch);

#endif // NCURSES_HELPERS_H
