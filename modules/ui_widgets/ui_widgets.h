#ifndef UI_WIDGETS_H
#define UI_WIDGETS_H

#include <stdint.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

// Create a button and check for mouse interaction
// Returns: 1 if clicked (mouse released over button), 0 otherwise
// Uses SDL mouse state internally
int64_t nl_ui_button(SDL_Renderer* renderer, TTF_Font* font,
                     const char* text, int64_t x, int64_t y, int64_t w, int64_t h);

// Draw a label (just text)
void nl_ui_label(SDL_Renderer* renderer, TTF_Font* font,
                 const char* text, int64_t x, int64_t y,
                 int64_t r, int64_t g, int64_t b, int64_t a);

// Slider widget - horizontal slider control
// Returns: new value (0.0 to 1.0)
// Uses SDL mouse state internally
double nl_ui_slider(SDL_Renderer* renderer, int64_t x, int64_t y, int64_t w, int64_t h,
                    double value);

// Draw a progress bar
void nl_ui_progress_bar(SDL_Renderer* renderer, int64_t x, int64_t y, int64_t w, int64_t h,
                        double progress);

#endif // UI_WIDGETS_H
