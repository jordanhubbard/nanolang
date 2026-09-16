#include <SDL2/SDL_mixer.h>

/* I retain SDL_mixer as an artifact dependency for exact-handle FFI lookup. */
int (*nl_sdl_mixer_link_anchor(void))(void) {
    return Mix_HaltMusic;
}
