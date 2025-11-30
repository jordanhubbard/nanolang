// SDL_ttf Helper Functions for Nanolang FFI
// Provides helper wrappers for common text rendering operations

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "sdl_ttf_helpers.h"

// Platform detection
#ifdef __APPLE__
#define PLATFORM_MACOS
#elif defined(__linux__)
#define PLATFORM_LINUX
#elif defined(_WIN32)
#define PLATFORM_WINDOWS
#endif

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

// Helper to open font with platform-specific fallback paths
// Tries multiple common font locations across platforms
// Returns font handle or 0 on failure
int64_t nl_open_font_portable(const char* font_name, int64_t ptsize) {
    TTF_Font* font = NULL;
    FILE* test_file = NULL;
    
    // List of font search paths for different platforms
    const char* search_paths[] = {
#ifdef PLATFORM_MACOS
        "/System/Library/Fonts/Supplemental/%s.ttf",
        "/System/Library/Fonts/%s.ttf",
        "/Library/Fonts/%s.ttf",
        "~/Library/Fonts/%s.ttf",
#endif
#ifdef PLATFORM_LINUX
        "/usr/share/fonts/truetype/dejavu/%s.ttf",
        "/usr/share/fonts/truetype/liberation/%s.ttf",
        "/usr/share/fonts/TTF/%s.ttf",
        "/usr/share/fonts/truetype/%s.ttf",
        "/usr/local/share/fonts/%s.ttf",
        "~/.fonts/%s.ttf",
#endif
#ifdef PLATFORM_WINDOWS
        "C:/Windows/Fonts/%s.ttf",
#endif
        // Generic fallback (relative path)
        "%s.ttf",
        NULL
    };
    
    // Build full paths and try each one
    char full_path[512];
    for (int i = 0; search_paths[i] != NULL; i++) {
        snprintf(full_path, sizeof(full_path), search_paths[i], font_name);
        
        // Check if file exists before trying to open
        test_file = fopen(full_path, "rb");
        if (test_file) {
            fclose(test_file);
            
            // File exists, try to open with SDL_ttf
            font = TTF_OpenFont(full_path, (int)ptsize);
            if (font) {
                return (int64_t)font;
            }
        }
    }
    
    // If nothing worked, try common font fallbacks
    const char* fallback_fonts[] = {
        "Arial",
        "DejaVuSans",
        "LiberationSans",
        "FreeSans",
        NULL
    };
    
    // Only try fallbacks if the requested font wasn't already a fallback
    int is_fallback = 0;
    for (int i = 0; fallback_fonts[i] != NULL; i++) {
        if (strcmp(font_name, fallback_fonts[i]) == 0) {
            is_fallback = 1;
            break;
        }
    }
    
    if (!is_fallback) {
        for (int i = 0; fallback_fonts[i] != NULL; i++) {
            font = (TTF_Font*)nl_open_font_portable(fallback_fonts[i], ptsize);
            if (font) {
                return (int64_t)font;
            }
        }
    }
    
    return 0;  // All attempts failed
}

