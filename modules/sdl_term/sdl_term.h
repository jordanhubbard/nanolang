#ifndef NL_SDL_TERM_H
#define NL_SDL_TERM_H

#include <stdint.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "tmt.h"

/* Opaque terminal handle. */
typedef struct NLTerm NLTerm;

/* Create a terminal backed by a libtmt virtual screen.
 * font_path  - path to a monospace .ttf font (NULL → platform default)
 * pt_size    - font point size (e.g. 14)
 * rows/cols  - terminal dimensions in characters
 * Returns opaque handle, or NULL on failure. */
NLTerm *nl_term_create(const char *font_path, int64_t pt_size,
                        int64_t rows, int64_t cols);

/* Feed bytes from the child process into the virtual terminal.
 * This updates the internal screen state; call nl_term_render to draw. */
void nl_term_feed(NLTerm *t, const char *data);

/* Render the virtual terminal into a SDL_Renderer at pixel position (x, y).
 * Only dirty lines are re-rendered; call nl_term_mark_clean afterward.
 * The renderer must already have a render target set. */
void nl_term_render(NLTerm *t, SDL_Renderer *renderer, int64_t x, int64_t y);

/* Mark the virtual screen clean (call after nl_term_render). */
void nl_term_mark_clean(NLTerm *t);

/* Resize the virtual terminal. */
void nl_term_resize(NLTerm *t, int64_t rows, int64_t cols);

/* Return cursor column (0-based). */
int64_t nl_term_cursor_col(NLTerm *t);

/* Return cursor row (0-based). */
int64_t nl_term_cursor_row(NLTerm *t);

/* Return the pixel width of one character cell. */
int64_t nl_term_cell_w(NLTerm *t);

/* Return the pixel height of one character cell. */
int64_t nl_term_cell_h(NLTerm *t);

/* Return total pixel width of the terminal (cols * cell_w). */
int64_t nl_term_pixel_w(NLTerm *t);

/* Return total pixel height of the terminal (rows * cell_h). */
int64_t nl_term_pixel_h(NLTerm *t);

/* Free all resources. */
void nl_term_destroy(NLTerm *t);

/* Convert an SDL_Keycode + modifier state into a terminal escape sequence.
 * Writes into out_buf (must be at least 16 bytes).
 * Returns the number of bytes to send, 0 if the key produces no output. */
int64_t nl_term_key_to_seq(int64_t sdl_keycode, int64_t sdl_mod,
                             char *out_buf, int64_t buf_len);

/* NanoLang-facing wrappers (all opaque handles passed as int64_t pointers) */

int64_t nl_sdl_term_create(const char *font_path, int64_t pt_size,
                             int64_t rows, int64_t cols);
void    nl_sdl_term_feed(int64_t handle, const char *data);
void    nl_sdl_term_render(int64_t handle, SDL_Renderer *renderer,
                            int64_t x, int64_t y);
void    nl_sdl_term_mark_clean(int64_t handle);
void    nl_sdl_term_resize(int64_t handle, int64_t rows, int64_t cols);
int64_t nl_sdl_term_cursor_col(int64_t handle);
int64_t nl_sdl_term_cursor_row(int64_t handle);
int64_t nl_sdl_term_cell_w(int64_t handle);
int64_t nl_sdl_term_cell_h(int64_t handle);
int64_t nl_sdl_term_pixel_w(int64_t handle);
int64_t nl_sdl_term_pixel_h(int64_t handle);
void    nl_sdl_term_destroy(int64_t handle);
int64_t nl_sdl_term_key_to_seq(int64_t sdl_keycode, int64_t sdl_mod,
                                 char *out_buf, int64_t buf_len);

#endif /* NL_SDL_TERM_H */
