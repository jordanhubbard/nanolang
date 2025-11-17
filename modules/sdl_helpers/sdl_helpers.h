#ifndef SDL_HELPERS_H
#define SDL_HELPERS_H

#include <stdint.h>

/* SDL Helper Functions - nanolang FFI bindings */

/* Render filled rectangle */
int64_t nl_sdl_render_fill_rect(int64_t renderer, int64_t x, int64_t y, int64_t w, int64_t h);

/* Poll for quit event */
int64_t nl_sdl_poll_event_quit(void);

/* Poll for mouse click */
int64_t nl_sdl_poll_mouse_click(void);

/* Poll for mouse state (holding) */
int64_t nl_sdl_poll_mouse_state(void);

/* Poll for keyboard events - returns scancode or -1 */
int64_t nl_sdl_poll_keypress(void);

/* Render text (solid - faster, no AA) */
int64_t nl_sdl_render_text_solid(int64_t renderer, int64_t font, const char* text, int64_t x, int64_t y, int64_t r, int64_t g, int64_t b, int64_t a);

/* Render text (blended - slower, anti-aliased) */
int64_t nl_sdl_render_text_blended(int64_t renderer, int64_t font, const char* text, int64_t x, int64_t y, int64_t r, int64_t g, int64_t b, int64_t a);

#endif /* SDL_HELPERS_H */
