#ifndef UI_WIDGETS_H
#define UI_WIDGETS_H

#include <stdint.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "../../src/runtime/dyn_array.h"

// Update widget mouse state - CALL THIS ONCE PER FRAME before rendering widgets!
// This allows all widgets to see the same mouse transition
void nl_ui_update_mouse_state();

// Set UI scale factor used for input hit-testing (mouse coordinates are divided by this value).
void nl_ui_set_scale(double scale);

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

// Checkbox widget - toggleable boolean state
// Returns: new state (1 for checked, 0 for unchecked)
// checked parameter is the current state
int64_t nl_ui_checkbox(SDL_Renderer* renderer, TTF_Font* font,
                       const char* label, int64_t x, int64_t y, int64_t checked);

// Radio button widget - selectable option in a group
// Returns: 1 if this button was clicked (should become selected), 0 otherwise
// selected parameter indicates if this button is currently selected
int64_t nl_ui_radio_button(SDL_Renderer* renderer, TTF_Font* font,
                           const char* label, int64_t x, int64_t y, int64_t selected);

// Panel widget - draws a styled container for grouping widgets
// Just draws the panel background/border - you place widgets inside manually
void nl_ui_panel(SDL_Renderer* renderer, int64_t x, int64_t y, int64_t w, int64_t h,
                 int64_t r, int64_t g, int64_t b, int64_t a);

// Scrollable list widget - displays a list of items with scrolling
// Returns index of clicked item, or -1 if none
int64_t nl_ui_scrollable_list(SDL_Renderer* renderer, TTF_Font* font,
                               DynArray* items, int64_t item_count,
                               int64_t x, int64_t y, int64_t w, int64_t h,
                               int64_t scroll_offset, int64_t selected_index);

// Time display widget - shows time in MM:SS format
void nl_ui_time_display(SDL_Renderer* renderer, TTF_Font* font,
                        int64_t seconds, int64_t x, int64_t y,
                        int64_t r, int64_t g, int64_t b, int64_t a);

// Seekable progress bar - interactive progress bar
// Returns new position if clicked, or -1.0 if not
double nl_ui_seekable_progress_bar(SDL_Renderer* renderer, int64_t x, int64_t y, int64_t w, int64_t h,
                                    double progress);

// Text input field - single line text input
// Returns 1 if Enter was pressed, 0 otherwise
// NOTE: Currently treated as read-only text for NanoLang integration.
int64_t nl_ui_text_input(SDL_Renderer* renderer, TTF_Font* font,
                          const char* buffer, int64_t buffer_size,
                          int64_t x, int64_t y, int64_t w, int64_t h,
                          int64_t is_focused);

// Dropdown/Combo box widget - shows selected item, expands when clicked
// Returns index of newly selected item, or -1 if no change
int64_t nl_ui_dropdown(SDL_Renderer* renderer, TTF_Font* font,
                       DynArray* items, int64_t item_count,
                       int64_t x, int64_t y, int64_t w, int64_t h,
                       int64_t selected_index, int64_t is_open);

// Number spinner widget - increment/decrement numeric value with +/- buttons
// Returns new value (clamped between min_val and max_val)
int64_t nl_ui_number_spinner(SDL_Renderer* renderer, TTF_Font* font,
                              int64_t value, int64_t min_val, int64_t max_val,
                              int64_t x, int64_t y, int64_t w, int64_t h);

// Tooltip widget - shows informational text when hovering over a widget area
// Call this after drawing the widget you want to add a tooltip to
void nl_ui_tooltip(SDL_Renderer* renderer, TTF_Font* font,
                   const char* text, int64_t widget_x, int64_t widget_y,
                   int64_t widget_w, int64_t widget_h);

// File selector widget - browse and select files from a directory
int64_t nl_ui_file_selector(SDL_Renderer* renderer, TTF_Font* font,
                             DynArray* files, int64_t file_count,
                             int64_t x, int64_t y, int64_t w, int64_t h,
                             int64_t scroll_offset, int64_t selected_index);

// Image button widget - clickable button with image texture
// Returns: 1 if button was clicked (mouse released over button), 0 otherwise
// Handles hover effect and click detection
// Parameters:
//   renderer: SDL renderer
//   texture_id: SDL texture ID (from SDL_image, cast to int64_t)
//   x, y: button position
//   w, h: button size (image will be scaled to fit)
//   hover_brightness: brightness multiplier on hover (1.0 = normal, 1.2 = 20% brighter)
int64_t nl_ui_image_button(SDL_Renderer* renderer, int64_t texture_id,
                             int64_t x, int64_t y, int64_t w, int64_t h,
                             double hover_brightness);

// Code display widget - syntax-highlighted code viewer
// Shows NanoLang source code with syntax highlighting and scrolling
// Parameters:
//   renderer: SDL renderer
//   font: TTF font for text rendering
//   code: source code string to display
//   x, y, w, h: display area rectangle
//   scroll_offset: number of lines to scroll from top
//   line_height: height of each line in pixels (typically font size + 4-6px)
void nl_ui_code_display(SDL_Renderer* renderer, TTF_Font* font,
                         const char* code, int64_t x, int64_t y,
                         int64_t w, int64_t h, int64_t scroll_offset,
                         int64_t line_height);

// Code display widget - ANSI-colored code viewer
// Expects ANSI color codes (e.g., "\x1b[35m") in the input string.
void nl_ui_code_display_ansi(SDL_Renderer* renderer, TTF_Font* font,
                             const char* code, int64_t x, int64_t y,
                             int64_t w, int64_t h, int64_t scroll_offset,
                             int64_t line_height);

// Code editor widget - syntax-highlighted code with line numbers and cursor
void nl_ui_code_editor(SDL_Renderer* renderer, TTF_Font* font,
                       const char* code, int64_t x, int64_t y,
                       int64_t w, int64_t h, int64_t scroll_offset,
                       int64_t line_height, int64_t cursor_row, int64_t cursor_col);

#endif // UI_WIDGETS_H
