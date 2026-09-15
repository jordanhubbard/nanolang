#include "sdl_image_helpers.h"

NANO_EXPORT_ARRAY_ABI(nl_img_load_icon_batch);
NANO_EXPORT_ARRAY_ABI(nl_img_destroy_texture_batch);
NANO_EXPORT_ARRAY_ABI(nl_img_get_supported_formats);

static bool array_prefix_valid(const DynArray *array, ElementType type,
                               size_t width, int64_t count) {
    return count >= 0 && (uint64_t)count <= SIZE_MAX / width &&
        dyn_array_has_storage(array, type, width, (uint64_t)count * width);
}

/* I allocate all result storage before acquiring any texture. */
DynArray *nl_img_load_icon_batch(SDL_Renderer *renderer, DynArray *files, int64_t count) {
    if (!renderer || !array_prefix_valid(files, ELEM_STRING, sizeof(char *), count))
        return NULL;
    DynArray *textures = dyn_array_new_with_capacity(ELEM_INT, count);
    if (!textures) return NULL;
    const char *const *names = files->data;
    int64_t *handles = textures->data;
    textures->length = count;
    for (int64_t i = 0; i < count; ++i) {
        SDL_Texture *texture = names[i] ? IMG_LoadTexture(renderer, names[i]) : NULL;
        handles[i] = (int64_t)(intptr_t)texture;
    }
    return textures;
}

/* I consume handles, not the runtime-owned array. Duplicate slots in this
 * same array are cleared together, including aliases outside the prefix.
 * Copies in another array are not ownership-safe aliases. */
void nl_img_destroy_texture_batch(DynArray *textures, int64_t count) {
    if (!array_prefix_valid(textures, ELEM_INT, sizeof(int64_t), count)) return;
    int64_t *handles = textures->data;
    for (int64_t i = 0; i < count; ++i) {
        int64_t handle = handles[i];
        if (!handle) continue;
        for (int64_t j = i; j < textures->length; ++j)
            if (handles[j] == handle) handles[j] = 0;
        SDL_DestroyTexture((SDL_Texture *)(intptr_t)handle);
    }
}

/* These names describe recognized extensions, not decoder availability in
 * every installed SDL_image build. Literal elements have static lifetime. */
DynArray *nl_img_get_supported_formats(void) {
    static const char *const formats[] = {
        "png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff",
        "webp", "pcx", "tga", "pnm", "xpm", "xcf", "svg"
    };
    const size_t count = sizeof formats / sizeof formats[0];
    DynArray *result = dyn_array_new_with_capacity(ELEM_STRING, (int64_t)count);
    if (!result) return NULL;
    for (size_t i = 0; i < count; ++i) dyn_array_push_string(result, formats[i]);
    return result;
}
