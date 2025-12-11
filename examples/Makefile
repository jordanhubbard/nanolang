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
#   onnx_*     - ONNX machine learning examples
#   sqlite_*   - SQLite database examples
#   curl_*     - HTTP/curl examples
#   uv_*       - libuv async I/O examples
#   event_*    - Event system examples
#   vector2d_* - Vector math examples
#   stdlib_*   - Standard library examples

BIN_DIR = ../bin
COMPILER = $(BIN_DIR)/nanoc
INTERPRETER = $(BIN_DIR)/nano

# Set module path for automatic module building
export NANO_MODULE_PATH=../modules

# === Nanolang-Only Examples (nl_*) ===
# These have no external dependencies, just pure nanolang

NL_EXAMPLES = \
	$(BIN_DIR)/nl_hello \
	$(BIN_DIR)/nl_calculator \
	$(BIN_DIR)/nl_factorial \
	$(BIN_DIR)/nl_fibonacci \
	$(BIN_DIR)/nl_primes \
	$(BIN_DIR)/nl_operators \
	$(BIN_DIR)/nl_strings \
	$(BIN_DIR)/nl_floats \
	$(BIN_DIR)/nl_loops \
	$(BIN_DIR)/nl_mutable \
	$(BIN_DIR)/nl_logical \
	$(BIN_DIR)/nl_comparisons \
	$(BIN_DIR)/nl_types \
	$(BIN_DIR)/nl_math \
	$(BIN_DIR)/nl_advanced_math \
	$(BIN_DIR)/nl_string_ops \
	$(BIN_DIR)/nl_arrays \
	$(BIN_DIR)/nl_array_complete \
	$(BIN_DIR)/nl_struct \
	$(BIN_DIR)/nl_enum \
	$(BIN_DIR)/nl_list_int \
	$(BIN_DIR)/nl_extern_math \
	$(BIN_DIR)/nl_extern_char \
	$(BIN_DIR)/nl_extern_string \
	$(BIN_DIR)/nl_random_sentence \
	$(BIN_DIR)/nl_pi_calculator \
	$(BIN_DIR)/nl_pi_simple \
	$(BIN_DIR)/nl_tictactoe \
	$(BIN_DIR)/nl_tracing \
	$(BIN_DIR)/nl_union_types \
	$(BIN_DIR)/nl_generic_lists \
	$(BIN_DIR)/nl_generic_list_basics \
	$(BIN_DIR)/nl_first_class_functions \
	$(BIN_DIR)/nl_filter_map_fold \
	$(BIN_DIR)/nl_function_factories \
	$(BIN_DIR)/nl_function_variables \
	$(BIN_DIR)/nl_matrix_operations \
	$(BIN_DIR)/nl_boids \
	$(BIN_DIR)/nl_game_of_life \
	$(BIN_DIR)/nl_snake \
	$(BIN_DIR)/nl_falling_sand \
	$(BIN_DIR)/nl_maze \
	$(BIN_DIR)/nl_generic_queue \
	$(BIN_DIR)/nl_generic_stack \
	$(BIN_DIR)/nl_for_loop_patterns \
	$(BIN_DIR)/nl_tuple_coordinates \
	$(BIN_DIR)/nl_language_features

# === SDL Examples (sdl_*) ===
# Require SDL2 library

SDL_EXAMPLES = \
	$(BIN_DIR)/sdl_pong \
	$(BIN_DIR)/sdl_fire \
	$(BIN_DIR)/sdl_starfield \
	$(BIN_DIR)/sdl_checkers \
	$(BIN_DIR)/sdl_boids \
	$(BIN_DIR)/sdl_particles \
	$(BIN_DIR)/sdl_raytracer \
	$(BIN_DIR)/sdl_asteroids \
	$(BIN_DIR)/sdl_terrain_explorer \
	$(BIN_DIR)/sdl_falling_sand \
	$(BIN_DIR)/sdl_nanoamp \
	$(BIN_DIR)/sdl_nanoviz \
	$(BIN_DIR)/sdl_mod_visualizer \
	$(BIN_DIR)/sdl_audio_player \
	$(BIN_DIR)/sdl_ui_widgets \
	$(BIN_DIR)/sdl_drawing_primitives \
	$(BIN_DIR)/sdl_texture_demo \
	$(BIN_DIR)/sdl_window_control \
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

# === Other Module Examples ===

ONNX_EXAMPLES = \
	$(BIN_DIR)/onnx_simple \
	$(BIN_DIR)/onnx_inference \
	$(BIN_DIR)/onnx_classifier

SQLITE_EXAMPLES = \
	$(BIN_DIR)/sqlite_simple

CURL_EXAMPLES = \
	$(BIN_DIR)/curl_example

UV_EXAMPLES = \
	$(BIN_DIR)/uv_example

EVENT_EXAMPLES = \
	$(BIN_DIR)/event_example

VECTOR2D_EXAMPLES = \
	$(BIN_DIR)/vector2d_demo

STDLIB_EXAMPLES = \
	$(BIN_DIR)/stdlib_ast_demo

# === All Examples ===

ALL_EXAMPLES = \
	$(NL_EXAMPLES) \
	$(SDL_EXAMPLES) \
	$(NCURSES_EXAMPLES) \
	$(OPENGL_EXAMPLES) \
	$(ONNX_EXAMPLES) \
	$(SQLITE_EXAMPLES) \
	$(CURL_EXAMPLES) \
	$(UV_EXAMPLES) \
	$(EVENT_EXAMPLES) \
	$(VECTOR2D_EXAMPLES) \
	$(STDLIB_EXAMPLES)

# === Targets ===

.PHONY: all nl sdl ncurses opengl onnx sqlite curl uv event vector2d stdlib clean help

# Default: build all examples
all: $(COMPILER) $(ALL_EXAMPLES)
	@echo ""
	@echo "All examples built successfully!"
	@echo ""
	@echo "Nanolang-only examples (nl_*):"
	@echo "  ./bin/nl_hello, nl_calculator, nl_factorial, nl_fibonacci, nl_primes..."
	@echo "  ./bin/nl_game_of_life, nl_snake, nl_maze, nl_boids, nl_falling_sand..."
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
	@echo "  ./bin/sdl_nanoamp           - Audio player with visualization"
	@echo "  ./bin/sdl_nanoviz           - Audio visualizer"
	@echo "  ./bin/sdl_mod_visualizer    - MOD music player"
	@echo ""
	@echo "NCurses examples (ncurses_*):"
	@echo "  ./bin/ncurses_snake         - Snake game"
	@echo "  ./bin/ncurses_game_of_life  - Conway's Game of Life"
	@echo "  ./bin/ncurses_matrix_rain   - Matrix digital rain"
	@echo ""
	@echo "OpenGL examples (opengl_*):"
	@echo "  ./bin/opengl_cube           - Rotating textured cube"
	@echo "  ./bin/opengl_teapot         - Rotating teapot"
	@echo ""

# Category targets
nl: $(COMPILER) $(NL_EXAMPLES)
sdl: $(COMPILER) $(SDL_EXAMPLES)
ncurses: $(COMPILER) $(NCURSES_EXAMPLES)
opengl: $(COMPILER) $(OPENGL_EXAMPLES)
onnx: $(COMPILER) $(ONNX_EXAMPLES)
sqlite: $(COMPILER) $(SQLITE_EXAMPLES)
curl: $(COMPILER) $(CURL_EXAMPLES)
uv: $(COMPILER) $(UV_EXAMPLES)
event: $(COMPILER) $(EVENT_EXAMPLES)
vector2d: $(COMPILER) $(VECTOR2D_EXAMPLES)
stdlib: $(COMPILER) $(STDLIB_EXAMPLES)

# Ensure compiler exists
$(COMPILER):
	@cd .. && $(MAKE) bin/nanoc

$(INTERPRETER):
	@cd .. && $(MAKE) bin/nano

# === Generic Build Rules ===

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

$(BIN_DIR)/opengl_teapot: opengl_teapot/teapot.nano $(COMPILER)
	@echo "Building OpenGL Teapot..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/opengl_teapot/teapot.nano -o bin/opengl_teapot

# ONNX examples
$(BIN_DIR)/onnx_%: onnx_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# SQLite examples
$(BIN_DIR)/sqlite_%: sqlite_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Curl examples
$(BIN_DIR)/curl_%: curl_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# UV examples
$(BIN_DIR)/uv_%: uv_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Event examples
$(BIN_DIR)/event_%: event_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Vector2D examples
$(BIN_DIR)/vector2d_%: vector2d_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Stdlib examples
$(BIN_DIR)/stdlib_%: stdlib_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && ./bin/nanoc examples/$< -o bin/$(notdir $@)

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
	@echo "  make              - Build all examples"
	@echo "  make nl           - Build nanolang-only examples"
	@echo "  make sdl          - Build SDL examples"
	@echo "  make ncurses      - Build NCurses examples"
	@echo "  make opengl       - Build OpenGL examples"
	@echo "  make onnx         - Build ONNX examples"
	@echo "  make sqlite       - Build SQLite examples"
	@echo "  make curl         - Build curl examples"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Naming Convention:"
	@echo "  nl_*       - Nanolang-only (no external deps)"
	@echo "  sdl_*      - SDL2 graphics"
	@echo "  ncurses_*  - NCurses terminal UI"
	@echo "  opengl_*   - OpenGL 3D graphics"
	@echo "  onnx_*     - ONNX machine learning"
	@echo "  sqlite_*   - SQLite database"
	@echo "  curl_*     - HTTP requests"
	@echo ""
