# examples/Makefile - Build all nanolang examples
#
# Usage:
#   make              - Build all compiled examples
#   make all          - Build all compiled examples
#   make sdl          - Build SDL examples only
#   make ncurses      - Build NCurses examples only
#   make nl           - Build nanolang-only examples
#   make opengl       - Build OpenGL examples only
#   make clean        - Remove build artifacts
#
# Naming Convention:
#   nl_*       - Nanolang-only examples (no external dependencies)
#   sdl_*      - SDL2 graphics examples
#   ncurses_*  - NCurses terminal UI examples
#   opengl_*   - OpenGL 3D graphics examples

BIN_DIR = ../bin
COMPILER = $(BIN_DIR)/nanoc
INTERPRETER = $(BIN_DIR)/nano

# Set module path for automatic module building
export NANO_MODULE_PATH=../modules

# === Nanolang-Only Compiled Examples (nl_*) ===
# After fixing the print statement bug, many more examples now compile!

NL_EXAMPLES = \
	$(BIN_DIR)/nl_hello \
	$(BIN_DIR)/nl_calculator \
	$(BIN_DIR)/nl_factorial \
	$(BIN_DIR)/nl_fibonacci \
	$(BIN_DIR)/nl_primes \
	$(BIN_DIR)/nl_enum \
	$(BIN_DIR)/nl_struct \
	$(BIN_DIR)/nl_comparisons \
	$(BIN_DIR)/nl_logical \
	$(BIN_DIR)/nl_operators \
	$(BIN_DIR)/nl_floats \
	$(BIN_DIR)/nl_types \
	$(BIN_DIR)/nl_mutable \
	$(BIN_DIR)/nl_union_types \
	$(BIN_DIR)/nl_extern_char \
	$(BIN_DIR)/nl_extern_math \
	$(BIN_DIR)/nl_extern_string \
	$(BIN_DIR)/nl_advanced_math \
	$(BIN_DIR)/nl_string_operations \
	$(BIN_DIR)/nl_matrix_operations \
	$(BIN_DIR)/nl_filter_map_fold \
	$(BIN_DIR)/nl_first_class_functions \
	$(BIN_DIR)/nl_function_variables \
	$(BIN_DIR)/nl_function_return_values \
	$(BIN_DIR)/nl_function_factories_v2 \
	$(BIN_DIR)/nl_array_bounds \
	$(BIN_DIR)/nl_array_complete \
	$(BIN_DIR)/nl_tracing \
	$(BIN_DIR)/nl_falling_sand \
	$(BIN_DIR)/nl_tictactoe \
	$(BIN_DIR)/nl_game_of_life \
	$(BIN_DIR)/nl_snake

# Temporarily disabled due to module import context bug (nanolang-6h9):
# - nl_demo_selfhosting (imports math_helper with 'as' alias)

# Derive interpreter-only list from current source set vs known-to-compile list.
NL_ALL_SOURCES := $(wildcard nl_*.nano)
NL_ALL_NAMES := $(sort $(basename $(NL_ALL_SOURCES)))
NL_COMPILED_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(NL_EXAMPLES)))
NL_INTERPRETER_ONLY_NAMES := $(sort $(filter-out $(NL_COMPILED_NAMES),$(NL_ALL_NAMES)))

# === SDL Examples (sdl_*) ===
# Require SDL2 library

SDL_EXAMPLES = \
	$(BIN_DIR)/sdl_pong \
	$(BIN_DIR)/sdl_fire \
	$(BIN_DIR)/sdl_starfield \
	$(BIN_DIR)/sdl_checkers \
	$(BIN_DIR)/sdl_boids \
	$(BIN_DIR)/sdl_particles \
	$(BIN_DIR)/sdl_mod_visualizer \
	$(BIN_DIR)/sdl_tracker_shell \
	$(BIN_DIR)/sdl_nanoamp \
	$(BIN_DIR)/sdl_raytracer \
	$(BIN_DIR)/sdl_asteroids \
	$(BIN_DIR)/sdl_terrain_explorer \
	$(BIN_DIR)/sdl_falling_sand \
	$(BIN_DIR)/sdl_ui_widgets_extended \
	$(BIN_DIR)/sdl_texture_demo \
	$(BIN_DIR)/sdl_drawing_primitives \
	$(BIN_DIR)/sdl_audio_wav \
	$(BIN_DIR)/sdl_mouse_click

# === NCurses Examples (ncurses_*) ===
# Require ncurses library

NCURSES_EXAMPLES = \
	$(BIN_DIR)/ncurses_snake \
	$(BIN_DIR)/ncurses_game_of_life \
	$(BIN_DIR)/ncurses_matrix_rain

# === OpenGL Examples (opengl_*) ===
# Require GLFW + GLEW

OPENGL_EXAMPLES = \
	$(BIN_DIR)/opengl_cube \
	$(BIN_DIR)/opengl_teapot

# === All Default Examples ===
# These are known to compile and work
# NOTE: OpenGL examples excluded by default (require GLFW+GLEW libraries)

ALL_EXAMPLES = \
	$(NL_EXAMPLES) \
	$(SDL_EXAMPLES) \
	$(NCURSES_EXAMPLES)

# === Targets ===

.PHONY: all build nl sdl ncurses opengl clean help launcher

# Default: build all examples
all: build

# Build all examples
build: $(COMPILER) $(ALL_EXAMPLES)
	@echo ""
	@echo "Default examples built successfully!"
	@echo ""
	@echo "Nanolang-only examples (nl_*) - $(words $(NL_COMPILED_NAMES)) compiled:"
	@$(if $(strip $(NL_COMPILED_NAMES)),$(foreach ex,$(NL_COMPILED_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "SDL examples (sdl_*):"
	@echo "  ./bin/sdl_pong              - Table tennis (1972)"
	@echo "  ./bin/sdl_asteroids         - Space shooter with HUD"
	@echo "  ./bin/sdl_checkers          - Board game with AI"
	@echo "  ./bin/sdl_fire              - Demoscene fire effect"
	@echo "  ./bin/sdl_starfield         - 3D star field"
	@echo "  ./bin/sdl_boids             - Flocking simulation"
	@echo "  ./bin/sdl_particles         - Particle effects"
	@echo "  ./bin/sdl_raytracer         - Real-time ray tracer"
	@echo "  ./bin/sdl_terrain_explorer  - 3D terrain"
	@echo "  ./bin/sdl_falling_sand      - Cellular automata sandbox"
	@echo "  ./bin/sdl_mod_visualizer    - MOD music player with visualizer"
	@echo "  ./bin/sdl_tracker_shell     - PT-style MOD player UI shell"
	@echo "  ./bin/sdl_nanoamp           - MP3 player (Winamp tribute)"
	@echo "  ./bin/sdl_ui_widgets_extended - Extended UI widgets demo (all widgets)"
	@echo "  ./bin/sdl_texture_demo      - Texture loading and rendering"
	@echo "  ./bin/sdl_drawing_primitives - Basic shape drawing"
	@echo "  ./bin/sdl_audio_wav         - Simple WAV audio playback"
	@echo "  ./bin/sdl_mouse_click       - Mouse input handling"
	@echo ""
	@echo "NCurses examples (ncurses_*):"
	@echo "  ./bin/ncurses_snake         - Snake game"
	@echo "  ./bin/ncurses_game_of_life  - Conway's Game of Life"
	@echo "  ./bin/ncurses_matrix_rain   - Matrix digital rain"
	@echo ""
	@echo "OpenGL examples (opengl_*) (not built by default):"
	@echo "  ./bin/opengl_cube           - Rotating textured cube (build: cd examples && make opengl)"
	@echo "  ./bin/opengl_teapot         - Utah teapot with multiple shaders (build: cd examples && make opengl)"
	@echo ""
	@echo "Other nl_* examples (not built by default):"
	@echo "  Run with interpreter: ./bin/nano examples/<file>"
	@echo "  Or try compiling individually: ./bin/nanoc examples/<file> -o bin/<name>"
	@echo "  $(words $(NL_INTERPRETER_ONLY_NAMES)) nl_* file(s) not built by default:"
	@$(if $(strip $(NL_INTERPRETER_ONLY_NAMES)),$(foreach ex,$(NL_INTERPRETER_ONLY_NAMES),echo "    - $(ex).nano";),:)
	@echo ""
	@echo "To launch the example browser:"
	@echo "  cd examples && make launcher"
	@echo "  # or from repo root: make examples-launcher"
	@echo ""

# Example launcher (SDL app-store style)
launcher: $(COMPILER)
	@echo "Building SDL example launcher..."
	@cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/sdl_example_launcher.nano -o bin/sdl_example_launcher
	@echo "Launching... (ESC or Quit to exit)"
	@cd .. && ./bin/sdl_example_launcher

# Category targets
nl: $(COMPILER) $(NL_EXAMPLES)
sdl: $(COMPILER) $(SDL_EXAMPLES)
ncurses: $(COMPILER) $(NCURSES_EXAMPLES)
opengl: $(COMPILER) $(OPENGL_EXAMPLES)

# Ensure compiler exists
$(COMPILER):
	@cd .. && $(MAKE) bin/nanoc

$(INTERPRETER):
	@cd .. && $(MAKE) bin/nano

# === Build Rules ===

# Nanolang-only examples (no module path needed)
$(BIN_DIR)/nl_%: nl_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && ./bin/nanoc examples/$< -o bin/$(notdir $@)

# SDL examples
$(BIN_DIR)/sdl_%: sdl_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# NCurses examples
$(BIN_DIR)/ncurses_%: ncurses_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# OpenGL examples
$(BIN_DIR)/opengl_cube: opengl_cube.nano $(COMPILER)
	@echo "Building OpenGL Cube..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/opengl_cube.nano -o bin/opengl_cube

$(BIN_DIR)/opengl_teapot: opengl_teapot.nano $(COMPILER)
	@echo "Building OpenGL Teapot..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/opengl_teapot.nano -o bin/opengl_teapot
	@echo "✓ OpenGL Teapot built: $@"

$(BIN_DIR)/sdl_nanoamp: sdl_nanoamp.nano $(COMPILER)
	@echo "Building SDL NanoAmp - Pixel-perfect Winamp tribute..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/sdl_nanoamp.nano -o bin/sdl_nanoamp
	@echo "✓ SDL NanoAmp built: $@"

$(BIN_DIR)/sdl_ui_widgets_extended: sdl_ui_widgets_extended.nano $(COMPILER)
	@echo "Building Extended UI Widgets Demo..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/sdl_ui_widgets_extended.nano -o bin/sdl_ui_widgets_extended
	@echo "✓ UI Widgets Extended built: $@"

# === Clean ===

clean:
	@echo "Cleaning example binaries..."
	rm -f $(ALL_EXAMPLES)
	rm -rf $(BIN_DIR)/*.dSYM
	rm -rf ../modules/*/.build/
	@echo "Done"

# === Help ===

help:
	@echo "Nanolang Examples Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build all examples (default)"
	@echo "  make build        - Build all examples"
	@echo "  make launcher     - Launch example browser (app store UI)"
	@echo "  make nl           - Build nanolang-only examples"
	@echo "  make sdl          - Build SDL examples"
	@echo "  make ncurses      - Build NCurses examples"
	@echo "  make opengl       - Build OpenGL examples"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Naming Convention:"
	@echo "  nl_*       - Nanolang-only (no external deps)"
	@echo "  sdl_*      - SDL2 graphics"
	@echo "  ncurses_*  - NCurses terminal UI"
	@echo "  opengl_*   - OpenGL 3D graphics"
	@echo ""
