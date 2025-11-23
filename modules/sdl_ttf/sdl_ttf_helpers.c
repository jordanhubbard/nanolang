// SDL_ttf Helper Functions for Nanolang FFI
// Provides helper wrappers for common text rendering operations

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Helper to render text to texture (blended, anti-aliased)
// Returns texture handle or 0 on failure
int64_t nl_render_text_blended_to_texture(int64_t renderer, int64_t font, const char* text, 
                                           int64_t r, int64_t g, int64_t b, int64_t a) {
    SDL_Color color = {(uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)a};
    SDL_Surface* surface = TTF_RenderText_Blended((TTF_Font*)font, text, color);
    
    if (!surface) {
        return 0;
    }
    
    SDL_Texture* texture = SDL_CreateTextureFromSurface((SDL_Renderer*)renderer, surface);
    SDL_FreeSurface(surface);
    
    return (int64_t)texture;
}

// Helper to draw text at position (blended, anti-aliased)
// Returns 1 on success, 0 on failure
int64_t nl_draw_text_blended(int64_t renderer, int64_t font, const char* text, 
                              int64_t x, int64_t y, int64_t r, int64_t g, int64_t b, int64_t a) {
    int64_t texture = nl_render_text_blended_to_texture(renderer, font, text, r, g, b, a);
    
    if (texture == 0) {
        return 0;
    }
    
    // Get texture dimensions
    int w, h;
    SDL_QueryTexture((SDL_Texture*)texture, NULL, NULL, &w, &h);
    
    // Render texture
    SDL_Rect src = {0, 0, w, h};
    SDL_Rect dst = {(int)x, (int)y, w, h};
    SDL_RenderCopy((SDL_Renderer*)renderer, (SDL_Texture*)texture, &src, &dst);
    
    SDL_DestroyTexture((SDL_Texture*)texture);
    return 1;
}

