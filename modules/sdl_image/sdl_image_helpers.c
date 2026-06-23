/*
 * SDL_image Helper Functions for NanoLang
 * Complete wrapper providing convenient functions for image loading and rendering
 */

#include "sdl_image_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ============================================================================
 * Basic Loading Helpers
 * ============================================================================ */

/* Load a PNG icon as a texture (convenience function) */
int64_t nl_img_load_png_texture(SDL_Renderer* renderer, const char* file) {
    if (!renderer || !file) {
        fprintf(stderr, "nl_img_load_png_texture: NULL parameter\n");
        return 0;
    }
    
    SDL_Texture* texture = IMG_LoadTexture(renderer, file);
    if (!texture) {
        fprintf(stderr, "nl_img_load_png_texture: Failed to load '%s': %s\n", 
                file, IMG_GetError());
        return 0;
    }
    
    return (int64_t)texture;
}

/* Load any image format as a texture with error handling */
int64_t nl_img_load_texture(SDL_Renderer* renderer, const char* file) {
    return nl_img_load_png_texture(renderer, file);  /* Same implementation */
}

/* ============================================================================
 * Rendering Helpers
 * ============================================================================ */

/* Render a texture at a specific position */
int64_t nl_img_render_texture(SDL_Renderer* renderer, int64_t texture,
                               int64_t x, int64_t y, int64_t w, int64_t h) {
    SDL_Texture* sdl_texture = (SDL_Texture*)texture;
    
    if (!renderer || !sdl_texture) {
        return -1;
    }
    
    SDL_Rect dest;
    dest.x = (int)x;
    dest.y = (int)y;
    
    /* If width and height are 0, use texture's original dimensions */
    if (w == 0 && h == 0) {
        if (SDL_QueryTexture(sdl_texture, NULL, NULL, &dest.w, &dest.h) != 0) {
            return -1;
        }
    } else {
        dest.w = (int)w;
        dest.h = (int)h;
    }
    
    if (SDL_RenderCopy(renderer, sdl_texture, NULL, &dest) != 0) {
        return -1;
    }
    
    return 0;
}

/* Render texture with rotation and flipping */
int64_t nl_img_render_texture_ex(SDL_Renderer* renderer, int64_t texture,
                                  int64_t x, int64_t y, int64_t w, int64_t h,
                                  double angle, int64_t flip) {
    SDL_Texture* sdl_texture = (SDL_Texture*)texture;
    
    if (!renderer || !sdl_texture) {
        return -1;
    }
    
    SDL_Rect dest;
    dest.x = (int)x;
    dest.y = (int)y;
    
    /* Get dimensions */
    if (w == 0 && h == 0) {
        if (SDL_QueryTexture(sdl_texture, NULL, NULL, &dest.w, &dest.h) != 0) {
            return -1;
        }
    } else {
        dest.w = (int)w;
        dest.h = (int)h;
    }
    
    /* Convert flip parameter to SDL_RendererFlip */
    SDL_RendererFlip sdl_flip = SDL_FLIP_NONE;
    if (flip == 1) sdl_flip = SDL_FLIP_HORIZONTAL;
    else if (flip == 2) sdl_flip = SDL_FLIP_VERTICAL;
    else if (flip == 3) sdl_flip = (SDL_RendererFlip)(SDL_FLIP_HORIZONTAL | SDL_FLIP_VERTICAL);
    
    if (SDL_RenderCopyEx(renderer, sdl_texture, NULL, &dest, angle, NULL, sdl_flip) != 0) {
        return -1;
    }
    
    return 0;
}

/* Render texture with source rectangle (sprite sheet support) */
int64_t nl_img_render_texture_sprite(SDL_Renderer* renderer, int64_t texture,
                                      int64_t src_x, int64_t src_y, int64_t src_w, int64_t src_h,
                                      int64_t dst_x, int64_t dst_y, int64_t dst_w, int64_t dst_h) {
    SDL_Texture* sdl_texture = (SDL_Texture*)texture;
    
    if (!renderer || !sdl_texture) {
        return -1;
    }
    
    SDL_Rect src = { (int)src_x, (int)src_y, (int)src_w, (int)src_h };
    SDL_Rect dst = { (int)dst_x, (int)dst_y, (int)dst_w, (int)dst_h };
    
    if (SDL_RenderCopy(renderer, sdl_texture, &src, &dst) != 0) {
        return -1;
    }
    
    return 0;
}

/* ============================================================================
 * Texture Info Helpers
 * ============================================================================ */

/* Get texture dimensions as a packed int64 */
int64_t nl_img_get_texture_size(int64_t texture) {
    SDL_Texture* sdl_texture = (SDL_Texture*)texture;
    
    if (!sdl_texture) {
        return 0;
    }
    
    int w, h;
    if (SDL_QueryTexture(sdl_texture, NULL, NULL, &w, &h) != 0) {
        return 0;
    }
    
    /* Pack width and height into a single int64_t */
    /* Width in high 32 bits, height in low 32 bits */
    return ((int64_t)w << 32) | (int64_t)h;
}

/* Get texture width only */
int64_t nl_img_get_texture_width(int64_t texture) {
    SDL_Texture* sdl_texture = (SDL_Texture*)texture;
    
    if (!sdl_texture) {
        return 0;
    }
    
    int w;
    if (SDL_QueryTexture(sdl_texture, NULL, NULL, &w, NULL) != 0) {
        return 0;
    }
    
    return (int64_t)w;
}

/* Get texture height only */
int64_t nl_img_get_texture_height(int64_t texture) {
    SDL_Texture* sdl_texture = (SDL_Texture*)texture;
    
    if (!sdl_texture) {
        return 0;
    }
    
    int h;
    if (SDL_QueryTexture(sdl_texture, NULL, NULL, NULL, &h) != 0) {
        return 0;
    }
    
    return (int64_t)h;
}

/* ============================================================================
 * Texture Property Helpers
 * ============================================================================ */

/* Set texture alpha modulation */
int64_t nl_img_set_texture_alpha(int64_t texture, int64_t alpha) {
    SDL_Texture* sdl_texture = (SDL_Texture*)texture;
    
    if (!sdl_texture) {
        return -1;
    }
    
    return (int64_t)SDL_SetTextureAlphaMod(sdl_texture, (Uint8)alpha);
}

/* Set texture color modulation */
int64_t nl_img_set_texture_color(int64_t texture, int64_t r, int64_t g, int64_t b) {
    SDL_Texture* sdl_texture = (SDL_Texture*)texture;
    
    if (!sdl_texture) {
        return -1;
    }
    
    return (int64_t)SDL_SetTextureColorMod(sdl_texture, (Uint8)r, (Uint8)g, (Uint8)b);
}

/* Set texture blend mode */
int64_t nl_img_set_texture_blend_mode(int64_t texture, int64_t blend) {
    SDL_Texture* sdl_texture = (SDL_Texture*)texture;
    
    if (!sdl_texture) {
        return -1;
    }
    
    SDL_BlendMode mode = SDL_BLENDMODE_NONE;
    switch (blend) {
        case 1: mode = SDL_BLENDMODE_BLEND; break;
        case 2: mode = SDL_BLENDMODE_ADD; break;
        case 4: mode = SDL_BLENDMODE_MOD; break;
        default: mode = SDL_BLENDMODE_NONE; break;
    }
    
    return (int64_t)SDL_SetTextureBlendMode(sdl_texture, mode);
}

/* ============================================================================
 * Advanced Creation
 * ============================================================================ */

/* Create a texture from pixel data */
int64_t nl_img_create_texture_from_pixels(SDL_Renderer* renderer, int64_t width, int64_t height, void* pixels) {
    if (!renderer || !pixels || width <= 0 || height <= 0) {
        return 0;
    }
    
    SDL_Surface* surface = SDL_CreateRGBSurfaceFrom(
        pixels,
        (int)width,
        (int)height,
        32,  /* 32 bits per pixel (RGBA) */
        (int)width * 4,  /* Pitch */
        0x000000FF,  /* R mask */
        0x0000FF00,  /* G mask */
        0x00FF0000,  /* B mask */
        0xFF000000   /* A mask */
    );
    
    if (!surface) {
        return 0;
    }
    
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FreeSurface(surface);
    
    return (int64_t)texture;
}

/* ============================================================================
 * Batch Operations
 * ============================================================================ */

/* Load multiple icons into an array (batch loading) */
void* nl_img_load_icon_batch(SDL_Renderer* renderer, const char** files, int64_t count) {
    if (!renderer || !files || count <= 0) {
        return NULL;
    }
    
    int64_t* textures = (int64_t*)malloc(sizeof(int64_t) * count);
    if (!textures) {
        return NULL;
    }
    
    for (int64_t i = 0; i < count; i++) {
        if (files[i]) {
            SDL_Texture* tex = IMG_LoadTexture(renderer, files[i]);
            textures[i] = (int64_t)tex;
            if (!tex) {
                fprintf(stderr, "nl_img_load_icon_batch: Failed to load '%s': %s\n",
                        files[i], IMG_GetError());
            }
        } else {
            textures[i] = 0;
        }
    }
    
    return (void*)textures;
}

/* Destroy multiple textures (batch cleanup) */
void nl_img_destroy_texture_batch(int64_t* textures, int64_t count) {
    if (!textures) {
        return;
    }
    
    for (int64_t i = 0; i < count; i++) {
        SDL_Texture* tex = (SDL_Texture*)textures[i];
        if (tex) {
            SDL_DestroyTexture(tex);
        }
    }
    
    free(textures);
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/* Destroy a texture */
void nl_img_destroy_texture(int64_t texture) {
    SDL_Texture* sdl_texture = (SDL_Texture*)texture;
    
    if (sdl_texture) {
        SDL_DestroyTexture(sdl_texture);
    }
}

/* Check if image file exists and is loadable */
int64_t nl_img_can_load(const char* file) {
    if (!file) {
        return 0;
    }
    
    /* Check if file exists */
    struct stat st;
    if (stat(file, &st) != 0) {
        return 0;
    }
    
    /* Check if it's a regular file */
    if (!S_ISREG(st.st_mode)) {
        return 0;
    }
    
    /* Try to detect format */
    SDL_RWops* rw = SDL_RWFromFile(file, "rb");
    if (!rw) {
        return 0;
    }
    
    /* Check various formats */
    int is_valid = 0;
    if (IMG_isPNG(rw) || IMG_isJPG(rw) || IMG_isBMP(rw) || 
        IMG_isGIF(rw) || IMG_isWEBP(rw) || IMG_isTIF(rw)) {
        is_valid = 1;
    }
    
    SDL_RWclose(rw);
    return (int64_t)is_valid;
}

/* Get supported image format extensions as array */
void* nl_img_get_supported_formats(void) {
    /* This would return a NanoLang array, but for now return a static list */
    static const char* formats[] = {
        "png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff",
        "webp", "pcx", "tga", "pnm", "xpm", "xcf", "svg",
        NULL
    };
    
    return (void*)formats;
}
