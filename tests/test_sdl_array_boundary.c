#include <SDL.h>
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include "runtime/dyn_array.h"

static int queries, updates, query_error;
static int texture_width = 2, texture_height = 1;
static Uint32 texture_format = SDL_PIXELFORMAT_ARGB8888;
static SDL_Texture *texture = (SDL_Texture *)(uintptr_t)1;
int SDLCALL SDL_QueryTexture(SDL_Texture *input, Uint32 *format, int *access, int *width, int *height) {
    assert(input == texture && access == NULL);
    ++queries;
    *format = texture_format;
    *width = texture_width;
    *height = texture_height;
    return query_error;
}
int SDLCALL SDL_UpdateTexture(SDL_Texture *input, const SDL_Rect *rect, const void *pixels, int pitch) {
    assert(input == texture && rect == NULL && pitch == 8);
    const uint32_t *values = pixels;
    assert(values[0] == 0x11223344 && values[1] == 0x55667788);
    ++updates;
    return 17;
}

/* My test runner copies this function verbatim from the production source. */
#include "sdl_update_under_test.c"

int main(void) {
    int64_t values[2] = {0x11223344, 0x55667788};
    DynArray array = {.length = 2, .capacity = 2, .elem_type = ELEM_INT,
                      .elem_size = sizeof(int64_t), .data = values};
    assert(nl_sdl_update_texture(texture, &array, 2, 1) == 17);
    assert(updates == 1 && queries == 1);
    assert(nl_sdl_update_texture(texture, &array, INT64_MAX, INT64_MAX) == -1);
    assert(nl_sdl_update_texture(texture, &array, 2, 0) == -1);
    assert(nl_sdl_update_texture(texture, &array, -1, 2) == -1);
    assert(nl_sdl_update_texture(texture, &array, 3, 1) == -1);
    assert(nl_sdl_update_texture(NULL, &array, 2, 1) == -1);
    array.elem_type = ELEM_FLOAT;
    assert(nl_sdl_update_texture(texture, &array, 2, 1) == -1);
    array.elem_type = ELEM_INT;
    array.elem_size = 1;
    assert(nl_sdl_update_texture(texture, &array, 2, 1) == -1);
    array.elem_size = sizeof(int64_t);
    array.capacity = 1;
    assert(nl_sdl_update_texture(texture, &array, 2, 1) == -1);
    array.capacity = 2;
    assert(queries == 1);
    texture_width = 3;
    assert(nl_sdl_update_texture(texture, &array, 2, 1) == -1);
    texture_width = 2;
    texture_height = 2;
    assert(nl_sdl_update_texture(texture, &array, 2, 1) == -1);
    texture_height = 1;
    texture_format = SDL_PIXELFORMAT_RGB565;
    assert(nl_sdl_update_texture(texture, &array, 2, 1) == -1);
    texture_format = SDL_PIXELFORMAT_NV12;
    assert(nl_sdl_update_texture(texture, &array, 2, 1) == -1);
    texture_format = SDL_PIXELFORMAT_ARGB8888;
    query_error = -1;
    assert(nl_sdl_update_texture(texture, &array, 2, 1) == -1);
    assert(updates == 1);
    return 0;
}
