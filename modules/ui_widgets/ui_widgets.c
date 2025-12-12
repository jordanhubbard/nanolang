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

// Draw a checkbox with label
// Returns new checked state (1 or 0)
int64_t nl_ui_checkbox(SDL_Renderer* renderer, TTF_Font* font,
                       const char* label, int64_t x, int64_t y, int64_t checked) {
    
    int64_t new_checked = checked;
    int box_size = 20;
    
    // Get mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, box_size, box_size);
    
    // Detect click
    if (prev_mouse_down && !mouse_down && is_hovered) {
        new_checked = !checked;
    }
    
    prev_mouse_down = mouse_down;
    
    // Choose colors based on state
    SDL_Color bg_color, border_color, check_color;
    
    if (is_hovered) {
        bg_color = (SDL_Color){100, 100, 120, 255};
        border_color = (SDL_Color){180, 180, 220, 255};
    } else {
        bg_color = (SDL_Color){60, 60, 80, 255};
        border_color = (SDL_Color){120, 120, 150, 255};
    }
    check_color = (SDL_Color){100, 200, 255, 255};
    
    // Draw checkbox background
    SDL_Rect box = {(int)x, (int)y, box_size, box_size};
    SDL_SetRenderDrawColor(renderer, bg_color.r, bg_color.g, bg_color.b, bg_color.a);
    SDL_RenderFillRect(renderer, &box);
    
    // Draw border
    SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
    SDL_RenderDrawRect(renderer, &box);
    
    // Draw check mark if checked
    if (new_checked) {
        SDL_SetRenderDrawColor(renderer, check_color.r, check_color.g, check_color.b, check_color.a);
        // Draw a simple X mark
        int padding = 4;
        SDL_RenderDrawLine(renderer, (int)x + padding, (int)y + padding, 
                          (int)x + box_size - padding, (int)y + box_size - padding);
        SDL_RenderDrawLine(renderer, (int)x + box_size - padding, (int)y + padding,
                          (int)x + padding, (int)y + box_size - padding);
        // Make it thicker
        SDL_RenderDrawLine(renderer, (int)x + padding + 1, (int)y + padding, 
                          (int)x + box_size - padding + 1, (int)y + box_size - padding);
        SDL_RenderDrawLine(renderer, (int)x + box_size - padding + 1, (int)y + padding,
                          (int)x + padding + 1, (int)y + box_size - padding);
    }
    
    // Draw label text
    if (font && label && strlen(label) > 0) {
        SDL_Color text_color = {220, 220, 220, 255};
        SDL_Surface* surface = TTF_RenderText_Blended(font, label, text_color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                int text_x = (int)x + box_size + 8;
                int text_y = (int)y + (box_size - surface->h) / 2;
                SDL_Rect dest = {text_x, text_y, surface->w, surface->h};
                SDL_RenderCopy(renderer, texture, NULL, &dest);
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
        }
    }
    
    return new_checked;
}

// Draw a radio button with label
// Returns 1 if clicked, 0 otherwise
int64_t nl_ui_radio_button(SDL_Renderer* renderer, TTF_Font* font,
                           const char* label, int64_t x, int64_t y, int64_t selected) {
    
    int clicked = 0;
    int circle_radius = 10;
    int circle_size = circle_radius * 2;
    
    // Get mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, circle_size, circle_size);
    
    // Detect click
    if (prev_mouse_down && !mouse_down && is_hovered) {
        clicked = 1;
    }
    
    prev_mouse_down = mouse_down;
    
    // Choose colors based on state
    SDL_Color bg_color, border_color, fill_color;
    
    if (is_hovered) {
        bg_color = (SDL_Color){100, 100, 120, 255};
        border_color = (SDL_Color){180, 180, 220, 255};
    } else {
        bg_color = (SDL_Color){60, 60, 80, 255};
        border_color = (SDL_Color){120, 120, 150, 255};
    }
    fill_color = (SDL_Color){100, 200, 255, 255};
    
    int center_x = (int)x + circle_radius;
    int center_y = (int)y + circle_radius;
    
    // Draw circle background (approximate with filled rect and pixels)
    SDL_Rect bg_rect = {(int)x, (int)y, circle_size, circle_size};
    SDL_SetRenderDrawColor(renderer, bg_color.r, bg_color.g, bg_color.b, bg_color.a);
    SDL_RenderFillRect(renderer, &bg_rect);
    
    // Draw circle border (approximate)
    SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
    for (int angle = 0; angle < 360; angle += 10) {
        double rad = angle * 3.14159 / 180.0;
        int px = center_x + (int)(circle_radius * cos(rad));
        int py = center_y + (int)(circle_radius * sin(rad));
        SDL_RenderDrawPoint(renderer, px, py);
        SDL_RenderDrawPoint(renderer, px + 1, py);
        SDL_RenderDrawPoint(renderer, px, py + 1);
    }
    
    // Draw filled circle if selected
    if (selected) {
        SDL_SetRenderDrawColor(renderer, fill_color.r, fill_color.g, fill_color.b, fill_color.a);
        int inner_radius = circle_radius - 4;
        // Fill inner circle
        for (int dy = -inner_radius; dy <= inner_radius; dy++) {
            for (int dx = -inner_radius; dx <= inner_radius; dx++) {
                if (dx * dx + dy * dy <= inner_radius * inner_radius) {
                    SDL_RenderDrawPoint(renderer, center_x + dx, center_y + dy);
                }
            }
        }
    }
    
    // Draw label text
    if (font && label && strlen(label) > 0) {
        SDL_Color text_color = {220, 220, 220, 255};
        SDL_Surface* surface = TTF_RenderText_Blended(font, label, text_color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                int text_x = (int)x + circle_size + 8;
                int text_y = (int)y + (circle_size - surface->h) / 2;
                SDL_Rect dest = {text_x, text_y, surface->w, surface->h};
                SDL_RenderCopy(renderer, texture, NULL, &dest);
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
        }
    }
    
    return clicked;
}

// Draw a panel (container for grouping widgets)
void nl_ui_panel(SDL_Renderer* renderer, int64_t x, int64_t y, int64_t w, int64_t h,
                 int64_t r, int64_t g, int64_t b, int64_t a) {
    
    // Draw background
    SDL_Rect bg = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, (uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)a);
    SDL_RenderFillRect(renderer, &bg);
    
    // Draw border (lighter than background)
    SDL_SetRenderDrawColor(renderer, 
                          (uint8_t)(r + 40 > 255 ? 255 : r + 40),
                          (uint8_t)(g + 40 > 255 ? 255 : g + 40),
                          (uint8_t)(b + 40 > 255 ? 255 : b + 40),
                          (uint8_t)a);
    SDL_RenderDrawRect(renderer, &bg);
}

// Scrollable list widget - displays a list of items with scrolling
// Returns index of clicked item, or -1 if none
int64_t nl_ui_scrollable_list(SDL_Renderer* renderer, TTF_Font* font,
                               nl_array_t* items, int64_t item_count,
                               int64_t x, int64_t y, int64_t w, int64_t h,
                               int64_t scroll_offset, int64_t selected_index) {
    
    if (!items || !font) return -1;
    
    int64_t clicked_index = -1;
    
    // Get mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    // Detect click
    int is_over_list = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, (int)w, (int)h);
    static int list_prev_mouse_down = 0;
    
    // Draw background
    SDL_Rect bg = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, 25, 25, 35, 255);
    SDL_RenderFillRect(renderer, &bg);
    
    // Draw border
    SDL_SetRenderDrawColor(renderer, 80, 80, 100, 255);
    SDL_RenderDrawRect(renderer, &bg);
    
    // Calculate visible items
    int item_height = 20;
    int visible_count = (int)h / item_height;
    
    // Draw items
    for (int i = 0; i < visible_count && (scroll_offset + i) < item_count; i++) {
        int64_t item_idx = scroll_offset + i;
        if (item_idx >= items->length) break;
        
        const char* item_text = (const char*)items->data[item_idx];
        if (!item_text) continue;
        
        int item_y = (int)y + (i * item_height);
        
        // Check if mouse is over this item
        int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, item_y, (int)w, item_height);
        
        // Detect click on this item
        if (list_prev_mouse_down && !mouse_down && is_hovered) {
            clicked_index = item_idx;
        }
        
        // Choose background color
        SDL_Color bg_color;
        if (item_idx == selected_index) {
            bg_color = (SDL_Color){60, 100, 180, 255};  // Selected
        } else if (is_hovered) {
            bg_color = (SDL_Color){50, 50, 70, 255};    // Hovered
        } else {
            bg_color = (SDL_Color){25, 25, 35, 255};    // Normal
        }
        
        // Draw item background
        SDL_Rect item_rect = {(int)x + 2, item_y, (int)w - 4, item_height};
        SDL_SetRenderDrawColor(renderer, bg_color.r, bg_color.g, bg_color.b, bg_color.a);
        SDL_RenderFillRect(renderer, &item_rect);
        
        // Draw item text
        SDL_Color text_color = {220, 220, 220, 255};
        SDL_Surface* surface = TTF_RenderText_Blended(font, item_text, text_color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                SDL_Rect dest = {(int)x + 5, item_y + 2, surface->w, surface->h};
                // Clip text if too wide
                if (dest.w > (int)w - 10) {
                    dest.w = (int)w - 10;
                }
                SDL_RenderCopy(renderer, texture, NULL, &dest);
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
        }
    }
    
    list_prev_mouse_down = mouse_down;
    return clicked_index;
}

// Time display widget - shows time in MM:SS format
void nl_ui_time_display(SDL_Renderer* renderer, TTF_Font* font,
                        int64_t seconds, int64_t x, int64_t y,
                        int64_t r, int64_t g, int64_t b, int64_t a) {
    
    if (!font) return;
    
    // Format time as MM:SS
    int minutes = (int)seconds / 60;
    int secs = (int)seconds % 60;
    
    char time_str[16];
    snprintf(time_str, sizeof(time_str), "%02d:%02d", minutes, secs);
    
    // Draw text
    SDL_Color color = {(uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)a};
    SDL_Surface* surface = TTF_RenderText_Blended(font, time_str, color);
    
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

// Seekable progress bar - interactive progress bar
// Returns new position if clicked, or -1.0 if not
double nl_ui_seekable_progress_bar(SDL_Renderer* renderer, int64_t x, int64_t y, int64_t w, int64_t h,
                                    double progress) {
    
    if (progress < 0.0) progress = 0.0;
    if (progress > 1.0) progress = 1.0;
    
    double new_position = -1.0;
    
    // Get mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    static int seek_prev_mouse_down = 0;
    int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, (int)w, (int)h);
    
    // Detect click and calculate new position
    if (seek_prev_mouse_down && !mouse_down && is_hovered) {
        new_position = (double)(mouse_x - x) / (double)w;
        if (new_position < 0.0) new_position = 0.0;
        if (new_position > 1.0) new_position = 1.0;
    }
    
    seek_prev_mouse_down = mouse_down;
    
    // Choose colors based on hover state
    SDL_Color bg_color = {40, 40, 50, 255};
    SDL_Color progress_color = is_hovered ? 
        (SDL_Color){100, 200, 120, 255} : 
        (SDL_Color){80, 180, 100, 255};
    SDL_Color border_color = is_hovered ?
        (SDL_Color){140, 140, 160, 255} :
        (SDL_Color){100, 100, 120, 255};
    
    // Draw background
    SDL_Rect bg = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, bg_color.r, bg_color.g, bg_color.b, bg_color.a);
    SDL_RenderFillRect(renderer, &bg);
    
    // Draw progress
    if (progress > 0.0) {
        SDL_Rect prog = {(int)x, (int)y, (int)(w * progress), (int)h};
        SDL_SetRenderDrawColor(renderer, progress_color.r, progress_color.g, progress_color.b, progress_color.a);
        SDL_RenderFillRect(renderer, &prog);
    }
    
    // Draw border
    SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
    SDL_RenderDrawRect(renderer, &bg);
    
    return new_position;
}
