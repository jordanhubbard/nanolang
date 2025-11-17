/*
 * SDL Helper Functions for nanolang
 * These functions wrap SDL struct operations that nanolang can't do directly
 */

#include <SDL.h>
#ifdef HAVE_SDL_TTF
#include <SDL_ttf.h>
#endif
#include <stdlib.h>
#include <stdint.h>

/* Helper to create SDL_Rect and call SDL_RenderFillRect */
/* Note: renderer is passed as int64_t (pointer value) and cast back to SDL_Renderer* */
int64_t nl_sdl_render_fill_rect(int64_t renderer_ptr, int64_t x, int64_t y, int64_t w, int64_t h) {
    SDL_Renderer *renderer = (SDL_Renderer*)renderer_ptr;
    SDL_Rect rect = {(int)x, (int)y, (int)w, (int)h};
    return SDL_RenderFillRect(renderer, &rect);
}

/* Helper to poll SDL events and return 1 if quit, 0 otherwise */
int64_t nl_sdl_poll_event_quit(void) {
    SDL_Event event;
    if (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            return 1;
        }
    }
    return 0;
}

/* Helper to poll for mouse button down events
 * Returns encoded position: row * 1000 + col * 100 + button, or -1 if no click
 * For checkers: we divide x,y by SQUARE_SIZE to get col,row
 * This version returns x * 10000 + y if left button clicked, -1 otherwise
 */
int64_t nl_sdl_poll_mouse_click(void) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            /* Store quit event for next poll_event_quit call */
            SDL_PushEvent(&event);
            return -1;
        }
        if (event.type == SDL_MOUSEBUTTONDOWN) {
            if (event.button.button == SDL_BUTTON_LEFT) {
                /* Encode x and y into single return value: x * 10000 + y */
                return (int64_t)event.button.x * 10000 + (int64_t)event.button.y;
            }
        }
    }
    return -1;
}

/* Helper to poll for mouse state (continuous holding)
 * Returns x * 10000 + y if left button is held, -1 otherwise
 */
int64_t nl_sdl_poll_mouse_state(void) {
    int x, y;
    Uint32 state = SDL_GetMouseState(&x, &y);
    if (state & SDL_BUTTON_LMASK) {
        return (int64_t)x * 10000 + (int64_t)y;
    }
    return -1;
}

/* Helper to poll for keyboard events
 * Returns SDL scancode if key pressed, -1 otherwise
 * Common scancodes:
 *   SPACE = 44, ESC = 41, C = 6
 *   0-9 = 30-39, 1 = 30, 2 = 31, etc.
 */
int64_t nl_sdl_poll_keypress(void) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            /* Store quit event for next poll_event_quit call */
            SDL_PushEvent(&event);
            return -1;
        }
        if (event.type == SDL_KEYDOWN) {
            return event.key.keysym.scancode;
        }
    }
    return -1;
}

#ifdef HAVE_SDL_TTF
/* Helper to render text using SDL_ttf
 * Creates SDL_Color struct, renders text, creates texture, and renders to screen
 * Returns 0 on success, -1 on failure
 */
int64_t nl_sdl_render_text_solid(int64_t renderer_ptr, int64_t font_ptr, 
                                  const char* text, int64_t x, int64_t y,
                                  int64_t r, int64_t g, int64_t b, int64_t a) {
    SDL_Renderer *renderer = (SDL_Renderer*)renderer_ptr;
    TTF_Font *font = (TTF_Font*)font_ptr;
    
    if (!font) return -1;
    
    SDL_Color color = {(Uint8)r, (Uint8)g, (Uint8)b, (Uint8)a};
    SDL_Surface *surface = TTF_RenderText_Solid(font, text, color);
    if (!surface) return -1;
    
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (!texture) {
        SDL_FreeSurface(surface);
        return -1;
    }
    
    SDL_Rect dest = {(int)x, (int)y, surface->w, surface->h};
    SDL_RenderCopy(renderer, texture, NULL, &dest);
    
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
    return 0;
}

/* Helper to render text using SDL_ttf with blended mode (anti-aliased)
 * Creates SDL_Color struct, renders text, creates texture, and renders to screen
 * Returns 0 on success, -1 on failure
 */
int64_t nl_sdl_render_text_blended(int64_t renderer_ptr, int64_t font_ptr, 
                                    const char* text, int64_t x, int64_t y,
                                    int64_t r, int64_t g, int64_t b, int64_t a) {
    SDL_Renderer *renderer = (SDL_Renderer*)renderer_ptr;
    TTF_Font *font = (TTF_Font*)font_ptr;
    
    if (!font) return -1;
    
    SDL_Color color = {(Uint8)r, (Uint8)g, (Uint8)b, (Uint8)a};
    SDL_Surface *surface = TTF_RenderText_Blended(font, text, color);
    if (!surface) return -1;
    
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (!texture) {
        SDL_FreeSurface(surface);
        return -1;
    }
    
    SDL_Rect dest = {(int)x, (int)y, surface->w, surface->h};
    SDL_RenderCopy(renderer, texture, NULL, &dest);
    
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
    return 0;
}
#endif /* HAVE_SDL_TTF */

