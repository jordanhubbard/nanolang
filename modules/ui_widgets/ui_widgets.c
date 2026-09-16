#include "ui_widgets.h"
#include <string.h>
#include <math.h>

// Helper: Check if point is inside rectangle
static int point_in_rect(int px, int py, int rx, int ry, int rw, int rh) {
    return px >= rx && px <= rx + rw && py >= ry && py <= ry + rh;
}

static double g_ui_scale = 1.0;

void nl_ui_set_scale(double scale) {
    if (scale <= 0.01) scale = 1.0;
    g_ui_scale = scale;
}

static void get_mouse_scaled(int *out_x, int *out_y) {
    int mx, my;
    SDL_GetMouseState(&mx, &my);
    mx = (int)((double)mx / g_ui_scale);
    my = (int)((double)my / g_ui_scale);
    if (out_x) *out_x = mx;
    if (out_y) *out_y = my;
}

// Static state for widget click detection (separate for each widget type)
// We track both CURRENT and PREVIOUS to detect transitions
static int button_current_mouse_down = 0;
static int button_prev_mouse_down = 0;
static int checkbox_current_mouse_down = 0;
static int checkbox_prev_mouse_down = 0;
static int radio_current_mouse_down = 0;
static int radio_prev_mouse_down = 0;

// Global text input buffer for SDL_TextInput events
static char g_text_input_buffer[256] = "";
static int g_text_input_active = 0;

// Helper: Start SDL text input mode
static void start_text_input() {
    if (!g_text_input_active) {
        SDL_StartTextInput();
        g_text_input_active = 1;
    }
}

// Helper: Stop SDL text input mode
static void stop_text_input() {
    if (g_text_input_active) {
        SDL_StopTextInput();
        g_text_input_active = 0;
    }
}

// Helper: Process SDL text input events and update buffer
// Call this from your main event loop
// Returns: 1 if text was modified, 0 otherwise
static int process_text_input_event(SDL_Event* event, char* buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) return 0;
    
    if (event->type == SDL_TEXTINPUT) {
        // Add new text to buffer
        size_t current_len = strlen(buffer);
        size_t input_len = strlen(event->text.text);
        
        if (current_len + input_len < buffer_size - 1) {
            strcat(buffer, event->text.text);
            return 1;
        }
    } else if (event->type == SDL_KEYDOWN) {
        if (event->key.keysym.sym == SDLK_BACKSPACE && strlen(buffer) > 0) {
            // Remove last character
            buffer[strlen(buffer) - 1] = '\0';
            return 1;
        } else if (event->key.keysym.sym == SDLK_RETURN || event->key.keysym.sym == SDLK_KP_ENTER) {
            // Enter pressed
            return 2;  // Special return value for Enter
        }
    }
    
    return 0;
}

// Update mouse state - call once per frame BEFORE rendering widgets
void nl_ui_update_mouse_state() {
    // Save current as previous
    button_prev_mouse_down = button_current_mouse_down;
    checkbox_prev_mouse_down = checkbox_current_mouse_down;
    radio_prev_mouse_down = radio_current_mouse_down;
    
    // Read new current state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    button_current_mouse_down = mouse_down;
    checkbox_current_mouse_down = mouse_down;
    radio_current_mouse_down = mouse_down;
}

// Draw a button with text
// Returns 1 if clicked, 0 otherwise
int64_t nl_ui_button(SDL_Renderer* renderer, TTF_Font* font,
                     const char* text, int64_t x, int64_t y, int64_t w, int64_t h) {
    
    // Get mouse position (state is tracked by nl_ui_update_mouse_state())
    int mouse_x, mouse_y;
    get_mouse_scaled(&mouse_x, &mouse_y);
    
    int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, (int)w, (int)h);
    int clicked = 0;
    
    // Detect click: was down prev frame, up current frame, and hovering
    if (button_prev_mouse_down && !button_current_mouse_down && is_hovered) {
        clicked = 1;
    }
    
    // State is updated once per frame by nl_ui_update_mouse_state(), not here!
    
    // Choose colors based on state
    SDL_Color bg_color, border_color, text_color;
    
    if (button_current_mouse_down && is_hovered) {
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
    SDL_Surface* surface = TTF_RenderUTF8_Blended(font, text, color);
    
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
    mouse_x = (int)((double)mouse_x / g_ui_scale);
    mouse_y = (int)((double)mouse_y / g_ui_scale);
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
    
    // Get mouse position (state tracked by nl_ui_update_mouse_state())
    int mouse_x, mouse_y;
    get_mouse_scaled(&mouse_x, &mouse_y);
    
    int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, box_size, box_size);
    
    // Detect click: was down prev frame, up current frame, and hovering
    if (checkbox_prev_mouse_down && !checkbox_current_mouse_down && is_hovered) {
        new_checked = !checked;
    }
    
    // State updated once per frame by nl_ui_update_mouse_state(), not here!
    
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
    
    // Get mouse position (state tracked by nl_ui_update_mouse_state())
    int mouse_x, mouse_y;
    get_mouse_scaled(&mouse_x, &mouse_y);
    
    int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, circle_size, circle_size);
    
    // Detect click: was down prev frame, up current frame, and hovering
    if (radio_prev_mouse_down && !radio_current_mouse_down && is_hovered) {
        clicked = 1;
    }
    
    // State updated once per frame by nl_ui_update_mouse_state(), not here!
    
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
                               DynArray* items, int64_t item_count,
                               int64_t x, int64_t y, int64_t w, int64_t h,
                               int64_t scroll_offset, int64_t selected_index) {
    
    if (!items || !font) return -1;
    
    int64_t clicked_index = -1;
    
    // Get mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    mouse_x = (int)((double)mouse_x / g_ui_scale);
    mouse_y = (int)((double)mouse_y / g_ui_scale);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    // Detect click
    int is_over_list = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, (int)w, (int)h);
    static int list_prev_mouse_down = 0;
    
    // Draw background
    SDL_Rect bg = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, 18, 18, 26, 255);
    SDL_RenderFillRect(renderer, &bg);
    
    // Draw border
    SDL_SetRenderDrawColor(renderer, 70, 70, 90, 255);
    SDL_RenderDrawRect(renderer, &bg);
    
    // Calculate visible items
    int item_height = 24;
    int visible_count = (int)h / item_height;

    // Clip list contents to its rectangle
    SDL_RenderSetClipRect(renderer, &bg);
    
    // Draw items
    for (int i = 0; i < visible_count && (scroll_offset + i) < item_count; i++) {
        int64_t item_idx = scroll_offset + i;
        if (item_idx >= items->length) break;
        
        const char* item_text = ((const char**)items->data)[item_idx];
        if (!item_text) continue;
        
        int item_y = (int)y + (i * item_height);
        
        // Check if mouse is over this item
        int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, item_y, (int)w, item_height);
        
        // Detect click on this item
        if (list_prev_mouse_down && !mouse_down && is_hovered) {
            clicked_index = item_idx;
        }
        
        // Choose background color
        SDL_Color row_even = (SDL_Color){22, 22, 30, 255};
        SDL_Color row_odd  = (SDL_Color){18, 18, 26, 255};
        SDL_Color bg_color;
        if (item_idx == selected_index) {
            bg_color = (SDL_Color){65, 110, 210, 255};  // Selected
        } else if (is_hovered) {
            bg_color = (SDL_Color){40, 40, 56, 255};    // Hovered
        } else {
            bg_color = ((i % 2) == 0) ? row_even : row_odd;
        }
        
        // Draw item background
        SDL_Rect item_rect = {(int)x + 2, item_y, (int)w - 4, item_height};
        SDL_SetRenderDrawColor(renderer, bg_color.r, bg_color.g, bg_color.b, bg_color.a);
        SDL_RenderFillRect(renderer, &item_rect);

        // Subtle separator
        SDL_SetRenderDrawColor(renderer, 28, 28, 38, 255);
        SDL_RenderDrawLine(renderer, item_rect.x, item_rect.y + item_rect.h - 1,
                           item_rect.x + item_rect.w, item_rect.y + item_rect.h - 1);

        // Selected accent bar
        if (item_idx == selected_index) {
            SDL_Rect accent = {item_rect.x, item_rect.y, 4, item_rect.h};
            SDL_SetRenderDrawColor(renderer, 140, 200, 255, 255);
            SDL_RenderFillRect(renderer, &accent);
        }
        
        // Draw item text
        SDL_Color text_color = (item_idx == selected_index) ? (SDL_Color){255, 255, 255, 255}
                                                            : (SDL_Color){225, 225, 235, 255};
        SDL_Surface* surface = TTF_RenderText_Blended(font, item_text, text_color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                int text_x = (int)x + 10;
                int text_y = item_y + (item_height - surface->h) / 2;
                SDL_Rect dest = {text_x, text_y, surface->w, surface->h};
                SDL_RenderCopy(renderer, texture, NULL, &dest);
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
        }
    }

    SDL_RenderSetClipRect(renderer, NULL);
    
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
    mouse_x = (int)((double)mouse_x / g_ui_scale);
    mouse_y = (int)((double)mouse_y / g_ui_scale);
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

// Text input field - single line text input
// Returns 1 if Enter was pressed, 0 otherwise
// Text buffer is modified in place
int64_t nl_ui_text_input(SDL_Renderer* renderer, TTF_Font* font,
                          const char* buffer, int64_t buffer_size,
                          int64_t x, int64_t y, int64_t w, int64_t h,
                          int64_t is_focused) {
    
    int enter_pressed = 0;
    
    // Get mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    mouse_x = (int)((double)mouse_x / g_ui_scale);
    mouse_y = (int)((double)mouse_y / g_ui_scale);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    static int input_prev_mouse_down = 0;
    int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, (int)w, (int)h);
    
    // Choose colors based on state
    SDL_Color bg_color, border_color, text_color;
    
    if (is_focused) {
        bg_color = (SDL_Color){70, 70, 90, 255};
        border_color = (SDL_Color){100, 150, 255, 255};
        text_color = (SDL_Color){255, 255, 255, 255};
    } else if (is_hovered) {
        bg_color = (SDL_Color){60, 60, 80, 255};
        border_color = (SDL_Color){140, 140, 180, 255};
        text_color = (SDL_Color){220, 220, 220, 255};
    } else {
        bg_color = (SDL_Color){50, 50, 70, 255};
        border_color = (SDL_Color){100, 100, 130, 255};
        text_color = (SDL_Color){200, 200, 200, 255};
    }
    
    // Draw background
    SDL_Rect rect = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, bg_color.r, bg_color.g, bg_color.b, bg_color.a);
    SDL_RenderFillRect(renderer, &rect);
    
    // Draw border (thicker if focused)
    SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
    SDL_RenderDrawRect(renderer, &rect);
    if (is_focused) {
        SDL_Rect inner = {(int)x + 1, (int)y + 1, (int)w - 2, (int)h - 2};
        SDL_RenderDrawRect(renderer, &inner);
    }
    
    // Draw text content
    if (font && buffer && strlen(buffer) > 0) {
        SDL_Surface* surface = TTF_RenderText_Blended(font, buffer, text_color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                int text_x = (int)x + 8;
                int text_y = (int)y + ((int)h - surface->h) / 2;
                SDL_Rect dest = {text_x, text_y, surface->w, surface->h};
                // Clip text if too wide
                if (dest.w > (int)w - 16) {
                    dest.w = (int)w - 16;
                }
                SDL_RenderCopy(renderer, texture, NULL, &dest);
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
        }
    }
    
    // Draw cursor if focused
    if (is_focused) {
        static int cursor_blink_counter = 0;
        cursor_blink_counter++;
        if ((cursor_blink_counter / 30) % 2 == 0) {  // Blink every 30 frames
            int cursor_x = (int)x + 8;
            if (buffer && strlen(buffer) > 0) {
                // Measure text width to position cursor
                int text_w, text_h;
                TTF_SizeText(font, buffer, &text_w, &text_h);
                cursor_x += text_w + 2;
            }
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            SDL_RenderDrawLine(renderer, cursor_x, (int)y + 6, cursor_x, (int)y + (int)h - 6);
        }
    }
    
    // Update mouse state only when it changes
    if (mouse_down != input_prev_mouse_down) {
        input_prev_mouse_down = mouse_down;
    }
    
    return enter_pressed;
}

// Dropdown/Combo box widget - shows selected item, expands to show options when clicked
// Returns: index of newly selected item, or -1 if no change
int64_t nl_ui_dropdown(SDL_Renderer* renderer, TTF_Font* font,
                       DynArray* items, int64_t item_count,
                       int64_t x, int64_t y, int64_t w, int64_t h,
                       int64_t selected_index, int64_t is_open) {
    
    if (!items || !font || item_count == 0) return -1;
    
    int64_t new_selection = -1;
    
    // Get mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    mouse_x = (int)((double)mouse_x / g_ui_scale);
    mouse_y = (int)((double)mouse_y / g_ui_scale);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    static int dropdown_prev_mouse_down = 0;
    int is_hovered = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, (int)w, (int)h);
    
    // Colors
    SDL_Color bg_color = is_hovered ? 
        (SDL_Color){80, 80, 100, 255} : 
        (SDL_Color){60, 60, 80, 255};
    SDL_Color border_color = {120, 120, 150, 255};
    SDL_Color text_color = {220, 220, 220, 255};
    SDL_Color arrow_color = {180, 180, 200, 255};
    
    // Draw main dropdown box
    SDL_Rect box = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, bg_color.r, bg_color.g, bg_color.b, bg_color.a);
    SDL_RenderFillRect(renderer, &box);
    SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
    SDL_RenderDrawRect(renderer, &box);
    
    // Draw selected item text
    if (selected_index >= 0 && selected_index < items->length) {
        const char* selected_text = ((const char**)items->data)[selected_index];
        if (selected_text) {
            SDL_Surface* surface = TTF_RenderText_Blended(font, selected_text, text_color);
            if (surface) {
                SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
                if (texture) {
                    int text_x = (int)x + 8;
                    int text_y = (int)y + ((int)h - surface->h) / 2;
                    SDL_Rect dest = {text_x, text_y, surface->w, surface->h};
                    if (dest.w > (int)w - 30) {
                        dest.w = (int)w - 30;
                    }
                    SDL_RenderCopy(renderer, texture, NULL, &dest);
                    SDL_DestroyTexture(texture);
                }
                SDL_FreeSurface(surface);
            }
        }
    }
    
    // Draw dropdown arrow
    int arrow_x = (int)x + (int)w - 20;
    int arrow_y = (int)y + (int)h / 2;
    SDL_SetRenderDrawColor(renderer, arrow_color.r, arrow_color.g, arrow_color.b, arrow_color.a);
    // Simple down arrow (triangle)
    for (int i = 0; i < 5; i++) {
        SDL_RenderDrawLine(renderer, arrow_x - i, arrow_y + i, arrow_x + i, arrow_y + i);
    }
    
    // Check for click on dropdown button to toggle open/close state
    if (dropdown_prev_mouse_down && !mouse_down && is_hovered && !is_open) {
        dropdown_prev_mouse_down = mouse_down;
        return -2;  // Signal to toggle dropdown open
    }
    
    // If dropdown is open, check for clicks outside to close it
    if (is_open && dropdown_prev_mouse_down && !mouse_down) {
        int list_h = (int)h * (item_count < 5 ? item_count : 5);
        int list_y = (int)y + (int)h;
        int in_dropdown_area = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, (int)w, (int)h + list_h);
        
        if (!in_dropdown_area) {
            dropdown_prev_mouse_down = mouse_down;
            return -3;  // Signal to close dropdown (clicked outside)
        }
    }
    
    // If dropdown is open, draw the list of options
    if (is_open && item_count > 0) {
        int item_h = (int)h;
        int list_h = (int)h * (item_count < 5 ? item_count : 5);
        int list_y = (int)y + (int)h;
        
        // Draw list background
        SDL_Rect list_bg = {(int)x, list_y, (int)w, list_h};
        SDL_SetRenderDrawColor(renderer, 45, 45, 65, 255);
        SDL_RenderFillRect(renderer, &list_bg);
        SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
        SDL_RenderDrawRect(renderer, &list_bg);
        
        // Draw items
        for (int i = 0; i < item_count && i < 5; i++) {
            const char* item_text = ((const char**)items->data)[i];
            if (!item_text) continue;
            
            int item_y = list_y + (i * item_h);
            int item_hovered = point_in_rect(mouse_x, mouse_y, (int)x, item_y, (int)w, item_h);
            
            // Highlight selected and hovered items
            if (i == selected_index) {
                SDL_Rect highlight = {(int)x + 2, item_y, (int)w - 4, item_h};
                SDL_SetRenderDrawColor(renderer, 60, 100, 180, 255);
                SDL_RenderFillRect(renderer, &highlight);
            } else if (item_hovered) {
                SDL_Rect highlight = {(int)x + 2, item_y, (int)w - 4, item_h};
                SDL_SetRenderDrawColor(renderer, 70, 70, 90, 255);
                SDL_RenderFillRect(renderer, &highlight);
            }
            
            // Check for click on this item
            if (dropdown_prev_mouse_down && !mouse_down && item_hovered) {
                new_selection = i;
            }
            
            // Draw item text
            SDL_Surface* surface = TTF_RenderText_Blended(font, item_text, text_color);
            if (surface) {
                SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
                if (texture) {
                    SDL_Rect dest = {(int)x + 8, item_y + (item_h - surface->h) / 2, surface->w, surface->h};
                    if (dest.w > (int)w - 16) {
                        dest.w = (int)w - 16;
                    }
                    SDL_RenderCopy(renderer, texture, NULL, &dest);
                    SDL_DestroyTexture(texture);
                }
                SDL_FreeSurface(surface);
            }
        }
    }
    
    // Update mouse state only when it changes
    if (mouse_down != dropdown_prev_mouse_down) {
        dropdown_prev_mouse_down = mouse_down;
    }
    
    return new_selection;
}

// Number spinner widget - increment/decrement numeric value
// Returns: new value
int64_t nl_ui_number_spinner(SDL_Renderer* renderer, TTF_Font* font,
                              int64_t value, int64_t min_val, int64_t max_val,
                              int64_t x, int64_t y, int64_t w, int64_t h) {
    
    int64_t new_value = value;
    
    // Get mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    mouse_x = (int)((double)mouse_x / g_ui_scale);
    mouse_y = (int)((double)mouse_y / g_ui_scale);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    static int spinner_prev_mouse_down = 0;
    
    int button_w = 20;
    SDL_Rect minus_btn = {(int)x, (int)y, button_w, (int)h};
    SDL_Rect plus_btn = {(int)x + (int)w - button_w, (int)y, button_w, (int)h};
    SDL_Rect value_area = {(int)x + button_w, (int)y, (int)w - 2 * button_w, (int)h};
    
    int minus_hovered = point_in_rect(mouse_x, mouse_y, minus_btn.x, minus_btn.y, minus_btn.w, minus_btn.h);
    int plus_hovered = point_in_rect(mouse_x, mouse_y, plus_btn.x, plus_btn.y, plus_btn.w, plus_btn.h);
    
    // Detect button clicks
    if (spinner_prev_mouse_down && !mouse_down) {
        if (minus_hovered && value > min_val) {
            new_value = value - 1;
        } else if (plus_hovered && value < max_val) {
            new_value = value + 1;
        }
    }
    
    // Colors
    SDL_Color minus_bg = minus_hovered ? (SDL_Color){90, 90, 110, 255} : (SDL_Color){70, 70, 90, 255};
    SDL_Color plus_bg = plus_hovered ? (SDL_Color){90, 90, 110, 255} : (SDL_Color){70, 70, 90, 255};
    SDL_Color border_color = {120, 120, 150, 255};
    SDL_Color text_color = {220, 220, 220, 255};
    
    // Draw minus button
    SDL_SetRenderDrawColor(renderer, minus_bg.r, minus_bg.g, minus_bg.b, minus_bg.a);
    SDL_RenderFillRect(renderer, &minus_btn);
    SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
    SDL_RenderDrawRect(renderer, &minus_btn);
    // Draw minus sign
    SDL_SetRenderDrawColor(renderer, text_color.r, text_color.g, text_color.b, text_color.a);
    int minus_y = (int)y + (int)h / 2;
    SDL_RenderDrawLine(renderer, (int)x + 5, minus_y, (int)x + button_w - 5, minus_y);
    
    // Draw plus button
    SDL_SetRenderDrawColor(renderer, plus_bg.r, plus_bg.g, plus_bg.b, plus_bg.a);
    SDL_RenderFillRect(renderer, &plus_btn);
    SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
    SDL_RenderDrawRect(renderer, &plus_btn);
    // Draw plus sign
    SDL_SetRenderDrawColor(renderer, text_color.r, text_color.g, text_color.b, text_color.a);
    int plus_x = plus_btn.x + button_w / 2;
    int plus_y = plus_btn.y + (int)h / 2;
    SDL_RenderDrawLine(renderer, plus_x - 5, plus_y, plus_x + 5, plus_y);
    SDL_RenderDrawLine(renderer, plus_x, plus_y - 5, plus_x, plus_y + 5);
    
    // Draw value area
    SDL_SetRenderDrawColor(renderer, 50, 50, 70, 255);
    SDL_RenderFillRect(renderer, &value_area);
    SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
    SDL_RenderDrawRect(renderer, &value_area);
    
    // Draw value text
    char value_str[32];
    snprintf(value_str, sizeof(value_str), "%lld", (long long)new_value);
    if (font) {
        SDL_Surface* surface = TTF_RenderText_Blended(font, value_str, text_color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                int text_x = value_area.x + (value_area.w - surface->w) / 2;
                int text_y = value_area.y + (value_area.h - surface->h) / 2;
                SDL_Rect dest = {text_x, text_y, surface->w, surface->h};
                SDL_RenderCopy(renderer, texture, NULL, &dest);
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
        }
    }
    
    // Update mouse state only when it changes
    if (mouse_down != spinner_prev_mouse_down) {
        spinner_prev_mouse_down = mouse_down;
    }
    
    return new_value;
}

// File selector widget - browse and select files
// Returns: selected file index, or -1 if no change
int64_t nl_ui_file_selector(SDL_Renderer* renderer, TTF_Font* font,
                             DynArray* files, int64_t file_count,
                             int64_t x, int64_t y, int64_t w, int64_t h,
                             int64_t scroll_offset, int64_t selected_index) {
    
    if (!files || !font || file_count == 0) return -1;
    
    int64_t clicked_item = -1;
    
    // Get mouse state
    int mouse_x, mouse_y;
    Uint32 mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    mouse_x = (int)((double)mouse_x / g_ui_scale);
    mouse_y = (int)((double)mouse_y / g_ui_scale);
    int mouse_down = (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    
    static int file_selector_prev_mouse_down = 0;
    
    // Draw background panel
    SDL_Rect bg = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, 40, 40, 50, 255);
    SDL_RenderFillRect(renderer, &bg);
    SDL_SetRenderDrawColor(renderer, 100, 100, 120, 255);
    SDL_RenderDrawRect(renderer, &bg);
    
    // Calculate visible items
    int item_h = 25;  // Height per file entry
    int visible_items = (int)h / item_h;
    int scroll_start = (int)scroll_offset;
    
    // Draw file list
    for (int i = scroll_start; i < file_count && (i - scroll_start) < visible_items; i++) {
        const char* filename = ((const char**)files->data)[i];
        if (!filename) continue;
        
        int item_y = (int)y + ((i - scroll_start) * item_h);
        int item_hovered = point_in_rect(mouse_x, mouse_y, (int)x, item_y, (int)w, item_h);
        
        // Highlight selected item
        if (i == selected_index) {
            SDL_Rect highlight = {(int)x + 2, item_y + 2, (int)w - 4, item_h - 2};
            SDL_SetRenderDrawColor(renderer, 60, 100, 180, 255);
            SDL_RenderFillRect(renderer, &highlight);
        } else if (item_hovered) {
            // Highlight hovered item
            SDL_Rect hover = {(int)x + 2, item_y + 2, (int)w - 4, item_h - 2};
            SDL_SetRenderDrawColor(renderer, 70, 70, 90, 255);
            SDL_RenderFillRect(renderer, &hover);
        }
        
        // Check for click
        if (file_selector_prev_mouse_down && !mouse_down && item_hovered) {
            clicked_item = i;
        }
        
        // Draw filename
        SDL_Color text_color = (i == selected_index) ? 
            (SDL_Color){255, 255, 255, 255} : 
            (SDL_Color){220, 220, 220, 255};
        
        SDL_Surface* surface = TTF_RenderText_Blended(font, filename, text_color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                SDL_Rect dest = {(int)x + 8, item_y + (item_h - surface->h) / 2, surface->w, surface->h};
                // Clip if too wide
                if (dest.w > (int)w - 16) {
                    dest.w = (int)w - 16;
                }
                SDL_RenderCopy(renderer, texture, NULL, &dest);
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
        }
    }
    
    // Update mouse state
    if (mouse_down != file_selector_prev_mouse_down) {
        file_selector_prev_mouse_down = mouse_down;
    }
    
    return clicked_item;
}

// Tooltip widget - shows informational text on hover
// Call this after drawing the widget you want to add a tooltip to
void nl_ui_tooltip(SDL_Renderer* renderer, TTF_Font* font,
                   const char* text, int64_t widget_x, int64_t widget_y,
                   int64_t widget_w, int64_t widget_h) {
    
    if (!font || !text || strlen(text) == 0) return;
    
    // Get mouse state
    int mouse_x, mouse_y;
    get_mouse_scaled(&mouse_x, &mouse_y);
    
    // Check if mouse is over the widget area
    if (!point_in_rect(mouse_x, mouse_y, (int)widget_x, (int)widget_y, (int)widget_w, (int)widget_h)) {
        return;  // Don't show tooltip if not hovering
    }
    
    // Measure text size
    int text_w, text_h;
    TTF_SizeText(font, text, &text_w, &text_h);
    
    int tooltip_w = text_w + 16;
    int tooltip_h = text_h + 12;
    int tooltip_x = mouse_x + 15;  // Offset from cursor
    int tooltip_y = mouse_y + 15;
    
    // Draw tooltip background
    SDL_Rect bg = {tooltip_x, tooltip_y, tooltip_w, tooltip_h};
    SDL_SetRenderDrawColor(renderer, 40, 40, 50, 240);
    SDL_RenderFillRect(renderer, &bg);
    
    // Draw border
    SDL_SetRenderDrawColor(renderer, 150, 150, 170, 255);
    SDL_RenderDrawRect(renderer, &bg);
    
    // Draw text
    SDL_Color text_color = {255, 255, 255, 255};
    SDL_Surface* surface = TTF_RenderText_Blended(font, text, text_color);
    if (surface) {
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
        if (texture) {
            SDL_Rect dest = {tooltip_x + 8, tooltip_y + 6, surface->w, surface->h};
            SDL_RenderCopy(renderer, texture, NULL, &dest);
            SDL_DestroyTexture(texture);
        }
        SDL_FreeSurface(surface);
    }
}

// Image button widget - clickable button with image texture
// Returns: 1 if button was clicked (mouse released over button), 0 otherwise
// Handles hover effect and click detection
//
// Parameters:
//   renderer: SDL renderer
//   texture_id: SDL texture ID (from SDL_image or similar, cast to int64_t)
//   x, y: button position
//   w, h: button size (image will be scaled to fit)
//   hover_brightness: brightness multiplier on hover (1.0 = no change, 1.2 = 20% brighter)
int64_t nl_ui_image_button(SDL_Renderer* renderer, int64_t texture_id,
                             int64_t x, int64_t y, int64_t w, int64_t h,
                             double hover_brightness) {
    
    SDL_Texture* texture = (SDL_Texture*)texture_id;
    if (!texture) return 0;  // Invalid texture
    
    // Get mouse state
    int mouse_x, mouse_y;
    get_mouse_scaled(&mouse_x, &mouse_y);
    
    // Check if mouse is over button
    int hover = point_in_rect(mouse_x, mouse_y, (int)x, (int)y, (int)w, (int)h);
    
    // Detect click: mouse was down last frame, up this frame, and over button
    int clicked = 0;
    if (button_prev_mouse_down && !button_current_mouse_down && hover) {
        clicked = 1;
    }
    
    // Draw the image texture
    SDL_Rect dest = {(int)x, (int)y, (int)w, (int)h};
    
    if (hover && hover_brightness > 1.0) {
        // Apply brightness modulation for hover effect
        Uint8 brightness = (Uint8)(255 * hover_brightness);
        if (brightness > 255) brightness = 255;
        SDL_SetTextureColorMod(texture, brightness, brightness, brightness);
    } else {
        // Normal brightness
        SDL_SetTextureColorMod(texture, 255, 255, 255);
    }
    
    SDL_RenderCopy(renderer, texture, NULL, &dest);
    
    // Draw border on hover
    if (hover) {
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 180);
        SDL_RenderDrawRect(renderer, &dest);
        
        // Draw inner highlight for more pronounced effect
        SDL_Rect inner = {(int)x + 1, (int)y + 1, (int)w - 2, (int)h - 2};
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 100);
        SDL_RenderDrawRect(renderer, &inner);
    }
    
    // Draw pressed effect
    if (button_current_mouse_down && hover) {
        // Darken slightly when pressed
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 40);
        SDL_RenderFillRect(renderer, &dest);
    }
    
    return clicked ? 1 : 0;
}

// ============================================================================
// Code Display Widget - Syntax-Highlighted Code Viewer
// ============================================================================

// Token types for syntax highlighting
typedef enum {
    TOKEN_KEYWORD,      // fn, let, if, while, etc.
    TOKEN_TYPE,         // int, string, bool, etc.
    TOKEN_STRING,       // String literals
    TOKEN_NUMBER,       // Numeric literals
    TOKEN_COMMENT,      // Comments
    TOKEN_OPERATOR,     // Operators: +, -, *, /, ==, etc.
    TOKEN_PAREN,        // Parentheses: ( )
    TOKEN_IDENTIFIER,   // Variables, functions
    TOKEN_WHITESPACE,   // Spaces, tabs
    TOKEN_TEXT          // Everything else
} TokenType;

// Token color scheme (RGB)
static void get_token_color(TokenType type, int* r, int* g, int* b) {
    switch (type) {
        case TOKEN_KEYWORD:
            *r = 220; *g = 120; *b = 255;  // Purple
            break;
        case TOKEN_TYPE:
            *r = 100; *g = 200; *b = 255;  // Cyan
            break;
        case TOKEN_STRING:
            *r = 255; *g = 200; *b = 100;  // Orange
            break;
        case TOKEN_NUMBER:
            *r = 150; *g = 255; *b = 150;  // Light green
            break;
        case TOKEN_COMMENT:
            *r = 120; *g = 120; *b = 120;  // Gray
            break;
        case TOKEN_OPERATOR:
            *r = 255; *g = 180; *b = 180;  // Light red
            break;
        case TOKEN_PAREN:
            *r = 200; *g = 200; *b = 100;  // Yellow
            break;
        case TOKEN_IDENTIFIER:
            *r = 220; *g = 220; *b = 240;  // Light gray
            break;
        default:
            *r = 200; *g = 200; *b = 200;  // Default gray
            break;
    }
}

// Check if character is part of an identifier
static int is_ident_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || 
           (c >= '0' && c <= '9') || c == '_';
}

// Check if string is a NanoLang keyword
static int is_keyword(const char* word, int len) {
    static const char* keywords[] = {
        "fn", "let", "mut", "if", "else", "while", "for", "return",
        "unsafe", "extern", "import", "from", "module", "struct", "enum",
        "pub", "assert", "shadow", "cond", "and", "or", "not", "set"
    };
    
    for (size_t i = 0; i < sizeof(keywords) / sizeof(keywords[0]); i++) {
        if (strlen(keywords[i]) == (size_t)len && 
            strncmp(word, keywords[i], len) == 0) {
            return 1;
        }
    }
    return 0;
}

// Check if string is a NanoLang type
static int is_type(const char* word, int len) {
    static const char* types[] = {
        "int", "float", "string", "bool", "void", "u8", "u16", "u32", "u64",
        "i8", "i16", "i32", "i64", "f32", "f64", "array"
    };
    
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        if (strlen(types[i]) == (size_t)len && 
            strncmp(word, types[i], len) == 0) {
            return 1;
        }
    }
    return 0;
}

// Simple tokenizer for NanoLang syntax
static TokenType get_token_type(const char* code, int pos, int* token_len) {
    char c = code[pos];
    
    // Whitespace
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        *token_len = 1;
        return TOKEN_WHITESPACE;
    }
    
    // Comments
    if (c == '#') {
        int len = 0;
        while (code[pos + len] && code[pos + len] != '\n') {
            len++;
        }
        *token_len = len;
        return TOKEN_COMMENT;
    }
    
    // String literals
    if (c == '"') {
        int len = 1;
        while (code[pos + len] && code[pos + len] != '"') {
            if (code[pos + len] == '\\' && code[pos + len + 1]) {
                len += 2;  // Skip escaped character
            } else {
                len++;
            }
        }
        if (code[pos + len] == '"') len++;  // Include closing quote
        *token_len = len;
        return TOKEN_STRING;
    }
    
    // Numbers
    if ((c >= '0' && c <= '9') || (c == '-' && code[pos+1] >= '0' && code[pos+1] <= '9')) {
        int len = (c == '-') ? 1 : 0;
        len++;
        while (code[pos + len] >= '0' && code[pos + len] <= '9') len++;
        if (code[pos + len] == '.') {
            len++;
            while (code[pos + len] >= '0' && code[pos + len] <= '9') len++;
        }
        *token_len = len;
        return TOKEN_NUMBER;
    }
    
    // Parentheses
    if (c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}') {
        *token_len = 1;
        return TOKEN_PAREN;
    }
    
    // Operators
    if (c == '+' || c == '-' || c == '*' || c == '/' || c == '%' ||
        c == '=' || c == '!' || c == '<' || c == '>' || c == ':' ||
        c == ',' || c == '.' || c == ';') {
        int len = 1;
        // Handle multi-character operators
        char next = code[pos + 1];
        if ((c == '=' && next == '=') || (c == '!' && next == '=') ||
            (c == '<' && next == '=') || (c == '>' && next == '=') ||
            (c == '-' && next == '>')) {
            len = 2;
        }
        *token_len = len;
        return TOKEN_OPERATOR;
    }
    
    // Identifiers and keywords
    if (is_ident_char(c) && !(c >= '0' && c <= '9')) {
        int len = 0;
        while (is_ident_char(code[pos + len])) len++;
        
        if (is_keyword(code + pos, len)) {
            *token_len = len;
            return TOKEN_KEYWORD;
        }
        if (is_type(code + pos, len)) {
            *token_len = len;
            return TOKEN_TYPE;
        }
        
        *token_len = len;
        return TOKEN_IDENTIFIER;
    }
    
    // Default: single character
    *token_len = 1;
    return TOKEN_TEXT;
}

// Code display widget - shows syntax-highlighted code with scrolling
// Parameters:
//   renderer: SDL renderer
//   font: TTF font for text
//   code: source code string to display
//   x, y, w, h: display area rectangle
//   scroll_offset: number of lines to scroll from top
//   line_height: height of each line in pixels (typically font size + padding)
void nl_ui_code_display(SDL_Renderer* renderer, TTF_Font* font,
                         const char* code, int64_t x, int64_t y, 
                         int64_t w, int64_t h, int64_t scroll_offset,
                         int64_t line_height) {
    
    if (!font || !code) return;
    
    // Draw background
    SDL_Rect bg = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, 20, 20, 28, 255);
    SDL_RenderFillRect(renderer, &bg);
    
    // Draw border
    SDL_SetRenderDrawColor(renderer, 60, 60, 70, 255);
    SDL_RenderDrawRect(renderer, &bg);
    
    // Set up clipping rectangle to prevent text overflow
    SDL_Rect clip = {(int)x + 2, (int)y + 2, (int)w - 4, (int)h - 4};
    SDL_RenderSetClipRect(renderer, &clip);
    
    int current_x = (int)x + 5;
    int current_y = (int)y + 5 - ((int)scroll_offset * (int)line_height);
    int pos = 0;
    int code_len = strlen(code);
    
    // Render tokens line by line
    while (pos < code_len) {
        char c = code[pos];
        
        // Handle newline
        if (c == '\n') {
            current_x = (int)x + 5;
            current_y += (int)line_height;
            pos++;
            
            // Early exit if we're past the visible area
            if (current_y > (int)y + (int)h) break;
            continue;
        }
        
        // Get token type and length
        int token_len = 0;
        TokenType token_type = get_token_type(code, pos, &token_len);
        
        if (token_len == 0) {
            pos++;
            continue;
        }
        
        // Skip rendering if above visible area
        if (current_y + (int)line_height < (int)y) {
            pos += token_len;
            continue;
        }
        
        // Skip whitespace rendering but advance position
        if (token_type == TOKEN_WHITESPACE) {
            if (c == ' ') {
                // Measure space width
                int space_w;
                TTF_SizeText(font, " ", &space_w, NULL);
                current_x += space_w;
            } else if (c == '\t') {
                int space_w;
                TTF_SizeText(font, " ", &space_w, NULL);
                current_x += space_w * 4;  // Tab = 4 spaces
            }
            pos++;
            continue;
        }
        
        // Extract token text
        char token_text[256];
        int copy_len = token_len < 255 ? token_len : 255;
        strncpy(token_text, code + pos, copy_len);
        token_text[copy_len] = '\0';
        
        // Get color for token type
        int r, g, b;
        get_token_color(token_type, &r, &g, &b);
        
        // Render token
        SDL_Color color = {(Uint8)r, (Uint8)g, (Uint8)b, 255};
        SDL_Surface* surface = TTF_RenderText_Blended(font, token_text, color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                SDL_Rect dest = {current_x, current_y, surface->w, surface->h};
                SDL_RenderCopy(renderer, texture, NULL, &dest);
                current_x += surface->w;
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
        }
        
        pos += token_len;
        
        // Wrap to next line if exceeds width
        if (current_x > (int)x + (int)w - 10) {
            current_x = (int)x + 5;
            current_y += (int)line_height;
            
            // Early exit if past visible area
            if (current_y > (int)y + (int)h) break;
        }
    }
    
    // Clear clipping
    SDL_RenderSetClipRect(renderer, NULL);
}

static void ansi_color_from_code(int code, int *r, int *g, int *b) {
    switch (code) {
        case 35: // keyword (purple)
            *r = 180; *g = 120; *b = 255;
            break;
        case 36: // type (cyan)
            *r = 100; *g = 220; *b = 255;
            break;
        case 32: // string (green)
            *r = 140; *g = 220; *b = 140;
            break;
        case 33: // number (yellow/orange)
            *r = 240; *g = 200; *b = 120;
            break;
        case 90: // comment (gray)
            *r = 120; *g = 120; *b = 140;
            break;
        case 37: // operator/paren (light gray)
            *r = 220; *g = 220; *b = 220;
            break;
        case 0:  // reset
        default:
            *r = 220; *g = 220; *b = 220;
            break;
    }
}

// Code display widget - ANSI-colored code viewer
void nl_ui_code_display_ansi(SDL_Renderer* renderer, TTF_Font* font,
                             const char* code, int64_t x, int64_t y,
                             int64_t w, int64_t h, int64_t scroll_offset,
                             int64_t line_height) {
    if (!font || !code) return;

    SDL_Rect bg = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, 20, 20, 28, 255);
    SDL_RenderFillRect(renderer, &bg);
    SDL_SetRenderDrawColor(renderer, 60, 60, 70, 255);
    SDL_RenderDrawRect(renderer, &bg);

    SDL_Rect clip = {(int)x + 2, (int)y + 2, (int)w - 4, (int)h - 4};
    SDL_RenderSetClipRect(renderer, &clip);

    int current_x = (int)x + 5;
    int current_y = (int)y + 5 - ((int)scroll_offset * (int)line_height);
    int pos = 0;
    int code_len = strlen(code);

    int r = 220, g = 220, b = 220;

    while (pos < code_len) {
        char c = code[pos];

        if (c == '\n') {
            current_x = (int)x + 5;
            current_y += (int)line_height;
            pos++;
            if (current_y > (int)y + (int)h) break;
            continue;
        }

        if (c == ' ') {
            int space_w;
            TTF_SizeText(font, " ", &space_w, NULL);
            current_x += space_w;
            pos++;
            continue;
        }

        if (c == '\t') {
            int space_w;
            TTF_SizeText(font, " ", &space_w, NULL);
            current_x += space_w * 4;
            pos++;
            continue;
        }

        if (c == 27 && code[pos + 1] == '[') {
            int seq = 0;
            int seq_pos = pos + 2;
            while (code[seq_pos] && code[seq_pos] != 'm') {
                if (code[seq_pos] >= '0' && code[seq_pos] <= '9') {
                    seq = seq * 10 + (code[seq_pos] - '0');
                }
                seq_pos++;
            }
            if (code[seq_pos] == 'm') {
                ansi_color_from_code(seq, &r, &g, &b);
                pos = seq_pos + 1;
                continue;
            }
        }

        // Render a run until whitespace/newline/escape
        char token_text[256];
        int token_len = 0;
        while (pos + token_len < code_len && token_len < 255) {
            char tc = code[pos + token_len];
            if (tc == '\n' || tc == ' ' || tc == '\t' || (tc == 27 && code[pos + token_len + 1] == '[')) {
                break;
            }
            token_text[token_len++] = tc;
        }
        if (token_len == 0) {
            pos++;
            continue;
        }
        token_text[token_len] = '\0';

        if (current_y + (int)line_height >= (int)y) {
            SDL_Color color = {(Uint8)r, (Uint8)g, (Uint8)b, 255};
            SDL_Surface* surface = TTF_RenderText_Blended(font, token_text, color);
            if (surface) {
                SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
                if (texture) {
                    SDL_Rect dst = {current_x, current_y, surface->w, surface->h};
                    SDL_RenderCopy(renderer, texture, NULL, &dst);
                    current_x += surface->w;
                    SDL_DestroyTexture(texture);
                }
                SDL_FreeSurface(surface);
            }
        }

        pos += token_len;
    }

    SDL_RenderSetClipRect(renderer, NULL);
}

// Code editor widget - displays code with line numbers, syntax highlighting, and blinking cursor
// Parameters same as nl_ui_code_display plus cursor_row and cursor_col (0-indexed)
void nl_ui_code_editor(SDL_Renderer* renderer, TTF_Font* font,
                       const char* code, int64_t x, int64_t y,
                       int64_t w, int64_t h, int64_t scroll_offset,
                       int64_t line_height, int64_t cursor_row, int64_t cursor_col) {
    if (!font || !code) return;

    // Background (slightly warmer to indicate edit mode)
    SDL_Rect bg = {(int)x, (int)y, (int)w, (int)h};
    SDL_SetRenderDrawColor(renderer, 25, 25, 35, 255);
    SDL_RenderFillRect(renderer, &bg);

    // Border (blue-ish for edit mode)
    SDL_SetRenderDrawColor(renderer, 70, 70, 120, 255);
    SDL_RenderDrawRect(renderer, &bg);

    // Clip rect
    SDL_Rect clip = {(int)x + 2, (int)y + 2, (int)w - 4, (int)h - 4};
    SDL_RenderSetClipRect(renderer, &clip);

    // Line number gutter
    int gutter_w = 50;

    // Gutter background
    SDL_Rect gutter_bg = {(int)x + 2, (int)y + 2, gutter_w - 2, (int)h - 4};
    SDL_SetRenderDrawColor(renderer, 30, 30, 42, 255);
    SDL_RenderFillRect(renderer, &gutter_bg);

    // Gutter separator
    SDL_SetRenderDrawColor(renderer, 60, 60, 80, 255);
    SDL_RenderDrawLine(renderer, (int)x + gutter_w, (int)y + 2,
                       (int)x + gutter_w, (int)y + (int)h - 2);

    int code_len = (int)strlen(code);
    int line_num = 0;
    int pos = 0;

    while (pos <= code_len) {
        // Find end of current line
        int line_start = pos;
        while (pos < code_len && code[pos] != '\n') pos++;
        int line_len = pos - line_start;

        // Y position for this line
        int draw_y = (int)y + 5 + (line_num - (int)scroll_offset) * (int)line_height;

        // Only render visible lines
        if (draw_y + (int)line_height >= (int)y && draw_y < (int)y + (int)h) {
            // Line number
            char line_num_str[16];
            snprintf(line_num_str, sizeof(line_num_str), "%3d", line_num + 1);
            SDL_Color gutter_color = {100, 100, 130, 255};
            if (line_num == (int)cursor_row) {
                gutter_color = (SDL_Color){180, 180, 220, 255};
            }
            SDL_Surface* gs = TTF_RenderText_Blended(font, line_num_str, gutter_color);
            if (gs) {
                SDL_Texture* gt = SDL_CreateTextureFromSurface(renderer, gs);
                if (gt) {
                    SDL_Rect dest = {(int)x + 5, draw_y, gs->w, gs->h};
                    SDL_RenderCopy(renderer, gt, NULL, &dest);
                    SDL_DestroyTexture(gt);
                }
                SDL_FreeSurface(gs);
            }

            // Highlight current line background
            if (line_num == (int)cursor_row) {
                SDL_SetRenderDrawColor(renderer, 35, 35, 50, 255);
                SDL_Rect line_bg = {(int)x + gutter_w + 2, draw_y,
                                    (int)w - gutter_w - 6, (int)line_height};
                SDL_RenderFillRect(renderer, &line_bg);
            }

            // Render line text with syntax highlighting
            if (line_len > 0) {
                int lpos = 0;
                int current_x = (int)x + gutter_w + 5;

                while (lpos < line_len) {
                    char c = code[line_start + lpos];

                    if (c == ' ') {
                        int sw;
                        TTF_SizeText(font, " ", &sw, NULL);
                        current_x += sw;
                        lpos++;
                        continue;
                    }
                    if (c == '\t') {
                        int sw;
                        TTF_SizeText(font, " ", &sw, NULL);
                        current_x += sw * 4;
                        lpos++;
                        continue;
                    }

                    // Tokenize from position in full code string
                    int token_len = 0;
                    TokenType token_type = get_token_type(code, line_start + lpos, &token_len);

                    if (token_len == 0) { lpos++; continue; }
                    if (token_type == TOKEN_WHITESPACE) {
                        lpos++;
                        continue;
                    }

                    // Clamp token to line boundary
                    if (lpos + token_len > line_len) {
                        token_len = line_len - lpos;
                    }
                    if (token_len <= 0) { lpos++; continue; }

                    char token_text[256];
                    int tcopy = token_len < 255 ? token_len : 255;
                    memcpy(token_text, code + line_start + lpos, tcopy);
                    token_text[tcopy] = '\0';

                    int tr, tg, tb;
                    get_token_color(token_type, &tr, &tg, &tb);

                    SDL_Color color = {(Uint8)tr, (Uint8)tg, (Uint8)tb, 255};
                    SDL_Surface* surf = TTF_RenderText_Blended(font, token_text, color);
                    if (surf) {
                        SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
                        if (tex) {
                            SDL_Rect dest = {current_x, draw_y, surf->w, surf->h};
                            SDL_RenderCopy(renderer, tex, NULL, &dest);
                            current_x += surf->w;
                            SDL_DestroyTexture(tex);
                        }
                        SDL_FreeSurface(surf);
                    }
                    lpos += token_len;
                }
            }

            // Draw blinking cursor
            if (line_num == (int)cursor_row) {
                int blink = ((SDL_GetTicks() / 530) % 2 == 0);
                if (blink) {
                    // Calculate cursor X by measuring prefix text (with tabs expanded)
                    int cursor_x = (int)x + gutter_w + 5;
                    if ((int)cursor_col > 0 && line_len > 0) {
                        int measure_len = (int)cursor_col < line_len ? (int)cursor_col : line_len;
                        char prefix[4096];
                        int pi = 0;
                        for (int ci = 0; ci < measure_len && pi < 4090; ci++) {
                            char ch = code[line_start + ci];
                            if (ch == '\t') {
                                prefix[pi++] = ' '; prefix[pi++] = ' ';
                                prefix[pi++] = ' '; prefix[pi++] = ' ';
                            } else {
                                prefix[pi++] = ch;
                            }
                        }
                        prefix[pi] = '\0';
                        int prefix_w;
                        TTF_SizeText(font, prefix, &prefix_w, NULL);
                        cursor_x += prefix_w;
                    }
                    SDL_SetRenderDrawColor(renderer, 255, 255, 100, 255);
                    SDL_Rect cursor_rect = {cursor_x, draw_y, 2, (int)line_height};
                    SDL_RenderFillRect(renderer, &cursor_rect);
                }
            }
        }

        line_num++;
        if (pos < code_len) pos++;  // skip newline
        else break;
    }

    SDL_RenderSetClipRect(renderer, NULL);
}
