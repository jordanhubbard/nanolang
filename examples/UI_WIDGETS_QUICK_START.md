# UI Widgets Quick Start Guide

## Getting Started

Import the UI widgets module in your nanolang SDL application:

```nano
import "modules/ui_widgets/ui_widgets.nano"
```

## Widget Reference

### Button
Click-responsive button with hover effects.

```nano
if (== (nl_ui_button renderer font "Click Me" 100 100 120 40) 1) {
    (println "Button clicked!")
} else {}
```

### Label
Static text display.

```nano
(nl_ui_label renderer font "Hello World" 10 10 255 255 255 255)
```

### Slider
Horizontal slider returning value 0.0-1.0.

```nano
set volume (nl_ui_slider renderer 50 50 200 20 volume)
```

### Progress Bar
Visual progress indicator (non-interactive).

```nano
(nl_ui_progress_bar renderer 50 50 300 25 0.75)  # 75% complete
```

### Checkbox ⭐ NEW
Toggleable on/off control.

```nano
set enabled (nl_ui_checkbox renderer font "Enable Feature" 50 50 enabled)

if (== enabled 1) {
    # Feature is enabled
} else {}
```

### Radio Button ⭐ NEW
Mutually exclusive option selection.

```nano
# Define the selected option
let mut selected_option: int = 0  # 0=Option A, 1=Option B, 2=Option C

# Radio button for Option A
let opt_a_selected: int = 0
if (== selected_option 0) { set opt_a_selected 1 } else {}
if (== (nl_ui_radio_button renderer font "Option A" 50 50 opt_a_selected) 1) {
    set selected_option 0
} else {}

# Radio button for Option B
let opt_b_selected: int = 0
if (== selected_option 1) { set opt_b_selected 1 } else {}
if (== (nl_ui_radio_button renderer font "Option B" 50 75 opt_b_selected) 1) {
    set selected_option 1
} else {}
```

### Panel ⭐ NEW
Visual grouping container for widgets.

```nano
# Draw panel first
(nl_ui_panel renderer 40 40 300 200 30 30 40 200)

# Then draw widgets inside
(nl_ui_label renderer font "Panel Title" 50 45 200 200 255 255)
(nl_ui_slider renderer 50 70 200 20 value)
```

## Common Patterns

### Radio Button Group Helper

```nano
# Helper pattern for cleaner radio groups
let option_1_sel: int = 0
if (== mode 1) { set option_1_sel 1 } else {}
if (== (nl_ui_radio_button renderer font "Mode 1" x y option_1_sel) 1) {
    set mode 1
} else {}
```

### Grouped Controls in Panel

```nano
# Draw panel background
(nl_ui_panel renderer 10 500 600 80 25 25 35 220)

# Add title
(nl_ui_label renderer font "Settings" 20 505 200 200 255 255)

# Add controls inside panel
set volume (nl_ui_slider renderer 20 530 200 20 volume)
set muted (nl_ui_checkbox renderer font "Mute" 250 530 muted)
```

### Conditional UI with Checkboxes

```nano
set show_advanced (nl_ui_checkbox renderer font "Advanced" 10 10 show_advanced)

if (== show_advanced 1) {
    # Draw additional controls
    (nl_ui_slider renderer 10 40 200 20 advanced_param)
} else {}
```

## Color Guidelines

### Standard Colors (matches existing examples)
- **Background Panel**: `(30, 30, 40, 200)` - Semi-transparent dark
- **Labels**: `(200, 200, 255, 255)` - Light blue-white for titles
- **Labels**: `(180, 180, 200, 255)` - Gray-white for descriptions
- **Success**: `(100, 255, 100, 255)` - Green
- **Warning**: `(255, 200, 0, 255)` - Orange
- **Error**: `(255, 100, 100, 255)` - Red

## Layout Tips

### Widget Spacing
- **Standard gap between widgets**: 10-15 pixels
- **Label to control gap**: 5 pixels
- **Panel padding**: 10 pixels on all sides
- **Checkbox/Radio height**: ~25 pixels (20px box + spacing)

### Typical Layout Pattern

```nano
let panel_x: int = 50
let panel_y: int = 400
let panel_width: int = 600
let panel_height: int = 150

(nl_ui_panel renderer panel_x panel_y panel_width panel_height 30 30 40 220)

# Title at top of panel
(nl_ui_label renderer font "Control Panel" (+ panel_x 10) (+ panel_y 5) 200 200 255 255)

# Controls with consistent spacing
let control_y: int = (+ panel_y 30)
(nl_ui_checkbox renderer font "Option 1" (+ panel_x 10) control_y checkbox1)
set control_y (+ control_y 25)
(nl_ui_checkbox renderer font "Option 2" (+ panel_x 10) control_y checkbox2)
set control_y (+ control_y 25)
(nl_ui_slider renderer (+ panel_x 10) control_y 200 20 slider_val)
```

## Complete Example

See `examples/ui_widgets_demo.nano` for a comprehensive demonstration of all widgets working together.

To run:
```bash
cd /Users/jkh/Src/nanolang
NANO_MODULE_PATH=modules ./bin/nanoc examples/ui_widgets_demo.nano -o bin/ui_widgets_demo
./bin/ui_widgets_demo
```

## Real-World Examples

Check these examples to see widgets in action:
- **`sdl_pong.nano`** - Checkbox for FPS display
- **`falling_sand_sdl.nano`** - Radio buttons for material selection in panel
- **`boids_sdl.nano`** - Panel grouping parameter sliders
- **`particles_sdl.nano`** - Panel for physics controls
- **`asteroids_complete_sdl.nano`** - Checkbox for debug mode

## Troubleshooting

### Radio Button Not Working?
Make sure you're converting boolean comparisons to int:
```nano
# WRONG - passes bool
(nl_ui_radio_button renderer font "A" 10 10 (== mode 0))

# RIGHT - passes int
let selected: int = 0
if (== mode 0) { set selected 1 } else {}
(nl_ui_radio_button renderer font "A" 10 10 selected)
```

### Widgets Not Responding to Clicks?
- Make sure you're calling widgets every frame
- Check that your render loop includes `SDL_RenderPresent`
- Verify event polling with `nl_sdl_poll_event_quit` and `nl_sdl_poll_keypress`

### Panel Not Showing?
- Draw panels BEFORE the widgets you want to appear on top
- Use semi-transparent alpha (150-220) to see content behind
- Ensure panel coordinates don't overlap the main rendering area

## Performance Notes

- All widgets are stateless and very lightweight
- Call widgets every frame (they handle their own state internally)
- No initialization required - just call the widget functions
- Mouse state is polled internally, no need to manage yourself

## Next Steps

Ready to build amazing UIs! For inspiration, check out **`nanoviz.nano`** - a 3D music visualizer that combines SDL, OpenGL, audio visualization, and UI widgets into one spectacular demo.
