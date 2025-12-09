#include "ui_widgets.h"
#include <string.h>
#include <math.h>

// Helper: Check if point is inside rectangle
static int point_in_rect(int px, int py, int rx, int ry, int rw, int rh) {
    return px >= rx && px <= rx + rw && py >= ry && py <= ry + rh;
}

// Static state for button click detection
static int prev_mouse_down = 0;

// Draw a button with text
// Returns 1 if clicked, 0 otherwise
int64_t nl_ui_button(SDL_Renderer* renderer, TTF_Font* font,
                     const char* text, int64_t x, int64_t y, int64_t w, int64_t h) {
    
    // Get current mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, (int)w, (int)h);
    int clicked = 0;
    
    // Detect click (mouse was down, now up, and cursor is over button)
    if (prev_mouse_down && !mouse_down && is_hovered) {
        clicked = 1;
    }
    
    // Update mouse state for next frame
    prev_mouse_down = mouse_down;
    
    // Choose colors based on state
    SDL_Color bg_color, border_color, text_color;
    
    if (mouse_down && is_hovered) {
        // Pressed
        bg_color = (SDL_Color){70, 70, 80, 255};
        border_color = (SDL_Color){150, 150, 200, 255};
        text_color = (SDL_Color){255, 255, 255, 255};
    } else if (is_hovered) {
        // Hovered
        bg_color = (SDL_Color){100, 100, 120, 255};
        border_color = (SDL_Color){180, 180, 220, 255};
        text_color = (SDL_Color){255, 255, 255, 255};
    } else {
        // Normal
        bg_color = (SDL_Color){80, 80, 100, 255};
        border_color = (SDL_Color){120, 120, 150, 255};
        text_color = (SDL_Color){220, 220, 220, 255};
    }
    
    // Draw button background
    SDL_Rect rect = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, bg_color.r, bg_color.g, bg_color.b, bg_color.a);
    SDL_RenderFillRect(renderer, &rect);
    
    // Draw button border
    SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
    SDL_RenderDrawRect(renderer, &rect);
    
    // Draw text centered in button
    if (font && text && strlen(text) > 0) {
        SDL_Surface* surface = TTF_RenderText_Blended(font, text, text_color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                int text_w = surface->w;
                int text_h = surface->h;
                
                // Center text
                int text_x = (int)x + ((int)w - text_w) / 2;
                int text_y = (int)y + ((int)h - text_h) / 2;
                
                SDL_Rect dest = {text_x, text_y, text_w, text_h};
                SDL_RenderCopy(renderer, texture, NULL, &dest);
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
        }
    }
    
    return clicked ? 1 : 0;
}

// Draw a label (text only)
void nl_ui_label(SDL_Renderer* renderer, TTF_Font* font,
                 const char* text, int64_t x, int64_t y,
                 int64_t r, int64_t g, int64_t b, int64_t a) {
    
    if (!font || !text || strlen(text) == 0) {
        return;
    }
    
    SDL_Color color = {(uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)a};
    SDL_Surface* surface = TTF_RenderText_Blended(font, text, color);
    
    if (surface) {
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
        if (texture) {
            SDL_Rect dest = {(int)x, (int)y, surface->w, surface->h};
            SDL_RenderCopy(renderer, texture, NULL, &dest);
            SDL_DestroyTexture(texture);
        }
        SDL_FreeSurface(surface);
    }
}

// Draw a horizontal slider
// Returns new value (0.0 to 1.0)
double nl_ui_slider(SDL_Renderer* renderer, int64_t x, int64_t y, int64_t w, int64_t h,
                    double value) {
    
    double new_value = value;
    
    // Get mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    // If mouse is down and over slider, update value
    if (mouse_down && point_in_rect(mouse_x, mouse_y, (int)x, (int)y, (int)w, (int)h)) {
        new_value = (double)(mouse_x - x) / (double)w;
        if (new_value < 0.0) new_value = 0.0;
        if (new_value > 1.0) new_value = 1.0;
    }
    
    // Draw slider track
    SDL_Rect track = {(int)x, (int)y + (int)h/3, (int)w, (int)h/3};
    SDL_SetRenderDrawColor(renderer, 60, 60, 70, 255);
    SDL_RenderFillRect(renderer, &track);
    
    // Draw filled portion
    if (new_value > 0.0) {
        SDL_Rect filled = {(int)x, (int)y + (int)h/3, (int)(w * new_value), (int)h/3};
        SDL_SetRenderDrawColor(renderer, 100, 150, 255, 255);
        SDL_RenderFillRect(renderer, &filled);
    }
    
    // Draw slider handle
    int handle_x = (int)(x + w * new_value);
    int handle_w = 8;
    SDL_Rect handle = {handle_x - handle_w/2, (int)y, handle_w, (int)h};
    SDL_SetRenderDrawColor(renderer, 180, 180, 200, 255);
    SDL_RenderFillRect(renderer, &handle);
    
    // Handle border
    SDL_SetRenderDrawColor(renderer, 220, 220, 240, 255);
    SDL_RenderDrawRect(renderer, &handle);
    
    return new_value;
}

// Draw a progress bar
void nl_ui_progress_bar(SDL_Renderer* renderer, int64_t x, int64_t y, int64_t w, int64_t h,
                        double progress) {
    
    if (progress < 0.0) progress = 0.0;
    if (progress > 1.0) progress = 1.0;
    
    // Draw background
    SDL_Rect bg = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, 40, 40, 50, 255);
    SDL_RenderFillRect(renderer, &bg);
    
    // Draw progress
    if (progress > 0.0) {
        SDL_Rect prog = {(int)x, (int)y, (int)(w * progress), (int)h};
        SDL_SetRenderDrawColor(renderer, 80, 180, 100, 255);
        SDL_RenderFillRect(renderer, &prog);
    }
    
    // Draw border
    SDL_SetRenderDrawColor(renderer, 100, 100, 120, 255);
    SDL_RenderDrawRect(renderer, &bg);
}
