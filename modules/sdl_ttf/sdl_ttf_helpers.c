// SDL_ttf Helper Functions for Nanolang FFI
// Provides wrappers for SDL_ttf font rendering functions

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// === Initialization ===

int64_t TTF_Init(void) {
    return (int64_t)TTF_Init();
}

int64_t TTF_Quit(void) {
    TTF_Quit();
    return 0;
}

int64_t TTF_WasInit(void) {
    return (int64_t)TTF_WasInit();
}

// === Font Management ===

int64_t TTF_OpenFont(const char* file, int64_t ptsize) {
    TTF_Font* font = TTF_OpenFont(file, (int)ptsize);
    return (int64_t)font;
}

int64_t TTF_CloseFont(int64_t font) {
    TTF_CloseFont((TTF_Font*)font);
    return 0;
}

// === Text Rendering ===

int64_t TTF_RenderText_Solid(int64_t font, const char* text, int64_t r, int64_t g, int64_t b, int64_t a) {
    SDL_Color color = {(uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)a};
    SDL_Surface* surface = TTF_RenderText_Solid((TTF_Font*)font, text, color);
    return (int64_t)surface;
}

int64_t TTF_RenderText_Blended(int64_t font, const char* text, int64_t r, int64_t g, int64_t b, int64_t a) {
    SDL_Color color = {(uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)a};
    SDL_Surface* surface = TTF_RenderText_Blended((TTF_Font*)font, text, color);
    return (int64_t)surface;
}

int64_t TTF_RenderText_Shaded(int64_t font, const char* text, 
                               int64_t fg_r, int64_t fg_g, int64_t fg_b, int64_t fg_a,
                               int64_t bg_r, int64_t bg_g, int64_t bg_b, int64_t bg_a) {
    SDL_Color fg_color = {(uint8_t)fg_r, (uint8_t)fg_g, (uint8_t)fg_b, (uint8_t)fg_a};
    SDL_Color bg_color = {(uint8_t)bg_r, (uint8_t)bg_g, (uint8_t)bg_b, (uint8_t)bg_a};
    SDL_Surface* surface = TTF_RenderText_Shaded((TTF_Font*)font, text, fg_color, bg_color);
    return (int64_t)surface;
}

// === Font Attributes ===

int64_t TTF_FontHeight(int64_t font) {
    return (int64_t)TTF_FontHeight((TTF_Font*)font);
}

int64_t TTF_FontAscent(int64_t font) {
    return (int64_t)TTF_FontAscent((TTF_Font*)font);
}

int64_t TTF_FontDescent(int64_t font) {
    return (int64_t)TTF_FontDescent((TTF_Font*)font);
}

int64_t TTF_FontLineSkip(int64_t font) {
    return (int64_t)TTF_FontLineSkip((TTF_Font*)font);
}

// === Text Metrics ===

int64_t TTF_SizeText(int64_t font, const char* text, int64_t w_out, int64_t h_out) {
    int w, h;
    int result = TTF_SizeText((TTF_Font*)font, text, &w, &h);
    if (w_out != 0) *(int*)w_out = w;
    if (h_out != 0) *(int*)h_out = h;
    return (int64_t)result;
}

// === Font Styles ===

int64_t TTF_GetFontStyle(int64_t font) {
    return (int64_t)TTF_GetFontStyle((TTF_Font*)font);
}

int64_t TTF_SetFontStyle(int64_t font, int64_t style) {
    TTF_SetFontStyle((TTF_Font*)font, (int)style);
    return 0;
}

// === Error Handling ===

const char* TTF_GetError(void) {
    return TTF_GetError();
}

int64_t TTF_ClearError(void) {
    SDL_ClearError();
    return 0;
}

