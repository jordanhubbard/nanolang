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
# These compile to native binaries (others are interpreter-only)

NL_EXAMPLES = \
	$(BIN_DIR)/nl_snake \
	$(BIN_DIR)/nl_game_of_life \
	$(BIN_DIR)/nl_falling_sand

# Note: Most nl_* examples are interpreter-only (run with: ../bin/nano <file>)
# They demonstrate language features but may use constructs the transpiler
# doesn't fully support yet.

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
	$(BIN_DIR)/sdl_raytracer \
	$(BIN_DIR)/sdl_asteroids \
	$(BIN_DIR)/sdl_terrain_explorer \
	$(BIN_DIR)/sdl_falling_sand

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

ALL_EXAMPLES = \
	$(NL_EXAMPLES) \
	$(SDL_EXAMPLES) \
	$(NCURSES_EXAMPLES) \
	$(OPENGL_EXAMPLES)

# === Targets ===

.PHONY: all nl sdl ncurses opengl clean help

# Default: build all examples
all: $(COMPILER) $(ALL_EXAMPLES)
	@echo ""
	@echo "All examples built successfully!"
	@echo ""
	@echo "Nanolang-only examples (nl_*):"
	@echo "  ./bin/nl_snake              - Snake game (terminal)"
	@echo "  ./bin/nl_game_of_life       - Conway's Game of Life (terminal)"
	@echo "  ./bin/nl_falling_sand       - Falling sand simulation (terminal)"
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
	@echo "Interpreter-only examples (run with: ./bin/nano examples/<file>):"
	@echo "  nl_hello, nl_calculator, nl_factorial, nl_fibonacci, nl_primes..."
	@echo ""

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

$(BIN_DIR)/opengl_teapot: opengl_teapot/teapot.nano $(COMPILER)
	@echo "Building OpenGL Teapot..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/opengl_teapot/teapot.nano -o bin/opengl_teapot

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
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Naming Convention:"
	@echo "  nl_*       - Nanolang-only (no external deps)"
	@echo "  sdl_*      - SDL2 graphics"
	@echo "  ncurses_*  - NCurses terminal UI"
	@echo "  opengl_*   - OpenGL 3D graphics"
	@echo ""
