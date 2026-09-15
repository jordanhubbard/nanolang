#include "../modules/sdl_image/sdl_image_helpers.h"
#include <assert.h>
#include <string.h>

static int loaded, destroyed;
static int texture_storage[4];
SDL_Texture *IMG_LoadTexture(SDL_Renderer *renderer, const char *name) {
    assert(renderer);
    ++loaded;
    if (!strcmp(name, "bad")) return NULL;
    return (SDL_Texture *)&texture_storage[loaded - 1];
}
void SDL_DestroyTexture(SDL_Texture *texture) {
    assert(texture);
    ++destroyed;
}

static int fail_allocation;
static DynArray *test_array_new(ElementType type, int64_t capacity) {
    return fail_allocation ? NULL : dyn_array_new_with_capacity(type, capacity);
}
#define dyn_array_new_with_capacity test_array_new
#include "../modules/sdl_image/sdl_image_arrays.c"
#undef dyn_array_new_with_capacity

int main(void) {
    gc_init();
    SDL_Renderer *renderer = (SDL_Renderer *)&texture_storage;
    DynArray *files = dyn_array_new(ELEM_STRING);
    assert(files);
    dyn_array_push_string(files, "good");
    dyn_array_push_string(files, "bad");
    dyn_array_push_string(files, NULL);
    dyn_array_push_string(files, "last");
    assert(!nl_img_load_icon_batch(NULL, files, 1));
    assert(!nl_img_load_icon_batch(renderer, NULL, 1));
    assert(!nl_img_load_icon_batch(renderer, files, -1));
    assert(!nl_img_load_icon_batch(renderer, files, 5));
    assert(!nl_img_load_icon_batch(renderer, files, INT64_MAX));
    DynArray invalid = *files;
    invalid.elem_type = ELEM_INT;
    assert(!nl_img_load_icon_batch(renderer, &invalid, 1));
    invalid = *files;
    invalid.elem_size = 1;
    assert(!nl_img_load_icon_batch(renderer, &invalid, 1));
    invalid = *files;
    invalid.capacity = 0;
    assert(!nl_img_load_icon_batch(renderer, &invalid, 1));
    fail_allocation = 1;
    assert(!nl_img_load_icon_batch(renderer, files, 4));
    assert(!nl_img_get_supported_formats());
    assert(loaded == 0);
    fail_allocation = 0;
    DynArray *empty = nl_img_load_icon_batch(renderer, files, 0);
    assert(empty && empty->length == 0 && loaded == 0);
    DynArray *textures = nl_img_load_icon_batch(renderer, files, 4);
    assert(textures && textures->elem_type == ELEM_INT && textures->length == 4);
    assert(loaded == 3);
    int64_t *handles = textures->data;
    assert(handles[0] && !handles[1] && !handles[2] && handles[3]);
    handles[1] = handles[0]; /* I clear duplicates even outside a requested prefix. */
    nl_img_destroy_texture_batch(textures, -1);
    nl_img_destroy_texture_batch(textures, 5);
    nl_img_destroy_texture_batch(files, 1);
    nl_img_destroy_texture_batch(NULL, 1);
    invalid = *textures;
    invalid.elem_size = 1;
    nl_img_destroy_texture_batch(&invalid, 1);
    invalid = *textures;
    invalid.length = -1;
    nl_img_destroy_texture_batch(&invalid, 1);
    invalid = *textures;
    invalid.capacity = 0;
    nl_img_destroy_texture_batch(&invalid, 1);
    assert(destroyed == 0);
    nl_img_destroy_texture_batch(textures, 1);
    assert(destroyed == 1 && !handles[0] && !handles[1] && handles[3]);
    nl_img_destroy_texture_batch(textures, 4);
    assert(destroyed == 2);
    nl_img_destroy_texture_batch(textures, 4);
    assert(destroyed == 2 && textures->length == 4);
    DynArray *formats = nl_img_get_supported_formats();
    assert(formats && formats->elem_type == ELEM_STRING && formats->length == 14);
    assert(!strcmp(dyn_array_get_string(formats, 0), "png"));
    assert(!strcmp(dyn_array_get_string(formats, 13), "svg"));
    gc_release(formats);
    gc_release(textures);
    gc_release(empty);
    gc_release(files);
    gc_shutdown();
    return 0;
}
