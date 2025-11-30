#ifndef SDL_TTF_HELPERS_H
#define SDL_TTF_HELPERS_H

#include <stdint.h>

/* SDL_ttf Helper Functions - nanolang FFI bindings */

/* Render text to texture (blended, anti-aliased) */
int64_t nl_render_text_blended_to_texture(int64_t renderer, int64_t font, const char* text, 
                                           int64_t r, int64_t g, int64_t b, int64_t a);

/* Draw text at position (blended, anti-aliased) */
int64_t nl_draw_text_blended(int64_t renderer, int64_t font, const char* text, 
                              int64_t x, int64_t y, int64_t r, int64_t g, int64_t b, int64_t a);

/* Open font with platform-specific fallback paths */
int64_t nl_open_font_portable(const char* font_name, int64_t ptsize);

#endif /* SDL_TTF_HELPERS_H */
