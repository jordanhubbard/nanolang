/*
 * SDL Helper Functions for nanolang
 * These functions wrap SDL struct operations that nanolang can't do directly
 */

#include <SDL.h>
#include <stdlib.h>
#include <stdint.h>

/* Helper to create SDL_Rect and call SDL_RenderFillRect */
int64_t nl_sdl_render_fill_rect(SDL_Renderer *renderer, int64_t x, int64_t y, int64_t w, int64_t h) {
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

