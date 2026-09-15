/* I retain SDL as a dependency of my manifest-bound foreign artifact. */
#include <SDL2/SDL.h>

int (*nl_sdl_library_anchor(void))(Uint32) {
    return SDL_Init;
}
