#ifndef SDL_IMAGE_HELPERS_H
#define SDL_IMAGE_HELPERS_H

#include <stdint.h>
#include <SDL.h>
#include <SDL_image.h>

/* Basic loading helpers */
int64_t nl_img_load_png_texture(SDL_Renderer* renderer, const char* file);
int64_t nl_img_load_texture(SDL_Renderer* renderer, const char* file);

/* Rendering helpers */
int64_t nl_img_render_texture(SDL_Renderer* renderer, int64_t texture, 
                               int64_t x, int64_t y, int64_t w, int64_t h);

int64_t nl_img_render_texture_ex(SDL_Renderer* renderer, int64_t texture,
                                  int64_t x, int64_t y, int64_t w, int64_t h,
                                  double angle, int64_t flip);

int64_t nl_img_render_texture_sprite(SDL_Renderer* renderer, int64_t texture,
                                      int64_t src_x, int64_t src_y, int64_t src_w, int64_t src_h,
                                      int64_t dst_x, int64_t dst_y, int64_t dst_w, int64_t dst_h);

/* Texture info helpers */
int64_t nl_img_get_texture_size(int64_t texture);
int64_t nl_img_get_texture_width(int64_t texture);
int64_t nl_img_get_texture_height(int64_t texture);

/* Texture property helpers */
int64_t nl_img_set_texture_alpha(int64_t texture, int64_t alpha);
int64_t nl_img_set_texture_color(int64_t texture, int64_t r, int64_t g, int64_t b);
int64_t nl_img_set_texture_blend_mode(int64_t texture, int64_t blend);

/* Advanced creation */
int64_t nl_img_create_texture_from_pixels(SDL_Renderer* renderer, int64_t width, int64_t height, void* pixels);

/* Batch operations */
void* nl_img_load_icon_batch(SDL_Renderer* renderer, const char** files, int64_t count);
void nl_img_destroy_texture_batch(int64_t* textures, int64_t count);

/* Utility functions */
void nl_img_destroy_texture(int64_t texture);
int64_t nl_img_can_load(const char* file);
void* nl_img_get_supported_formats(void);

#endif /* SDL_IMAGE_HELPERS_H */
