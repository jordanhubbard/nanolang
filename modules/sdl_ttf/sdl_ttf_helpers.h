#ifndef SDL_TTF_HELPERS_H
#define SDL_TTF_HELPERS_H

#include <stdint.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

/* SDL_ttf Helper Functions - nanolang FFI bindings */

/* Render text to texture (blended, anti-aliased) */
int64_t nl_render_text_blended_to_texture(SDL_Renderer* renderer, TTF_Font* font, const char* text, 
                                           int64_t r, int64_t g, int64_t b, int64_t a);

/* Draw text at position (blended, anti-aliased) */
int64_t nl_draw_text_blended(SDL_Renderer* renderer, TTF_Font* font, const char* text, 
                              int64_t x, int64_t y, int64_t r, int64_t g, int64_t b, int64_t a);

/* Open font with platform-specific fallback paths */
TTF_Font* nl_open_font_portable(const char* font_name, int64_t ptsize);

#endif /* SDL_TTF_HELPERS_H */
