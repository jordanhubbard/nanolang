# examples/Makefile - Build all nanolang examples
#
# Usage:
#   make              - Build all compiled examples
#   make all          - Build all compiled examples
#   make sdl          - Build SDL examples only
#   make interpreter  - Run all interpreter-only examples
#   make checkers     - Build checkers example
#   make boids-sdl    - Build visual boids simulation
#   make clean        - Remove build artifacts
#
# Note: With the automatic module system, just run nanoc and it handles everything!
#       Examples with advanced features (struct arrays) are interpreter-only for now.

BIN_DIR = ../bin
COMPILER = $(BIN_DIR)/nanoc
INTERPRETER = $(BIN_DIR)/nano

# Set module path for automatic module building
export NANO_MODULE_PATH=../modules

# === Compiled Examples (Transpiler) ===

# SDL examples that compile successfully
SDL_EXAMPLES = \
	$(BIN_DIR)/checkers_sdl \
	$(BIN_DIR)/boids_sdl \
	$(BIN_DIR)/particles_sdl \
	$(BIN_DIR)/falling_sand_sdl \
	$(BIN_DIR)/terrain_explorer_sdl

# OpenGL examples (GLFW + GLEW)
OPENGL_EXAMPLES = \
	$(BIN_DIR)/opengl_cube \
	$(BIN_DIR)/opengl_teapot

# All graphics examples
GRAPHICS_EXAMPLES = $(SDL_EXAMPLES) $(OPENGL_EXAMPLES)

# === Interpreter-Only Examples ===
# These use features the transpiler doesn't support yet
# Run with: ../bin/nano examples/name.nano

INTERPRETER_EXAMPLES = \
	life_sdl.nano \
	snake_sdl.nano \
	maze_sdl.nano

# === Targets ===

.PHONY: all sdl opengl interpreter clean help checkers-sdl boids-sdl particles-sdl falling-sand-sdl terrain-explorer-sdl opengl-cube opengl-teapot run-% force-rebuild

# Default: build all compiled examples
all: $(COMPILER) $(GRAPHICS_EXAMPLES)
	@echo ""
	@echo "✓ All compiled examples built successfully!"
	@echo ""
	@echo "SDL examples:"
	@echo "  ./bin/checkers_sdl"
	@echo "  ./bin/boids_sdl"
	@echo "  ./bin/particles_sdl"
	@echo "  ./bin/falling_sand_sdl"
	@echo "  ./bin/terrain_explorer_sdl"
	@echo ""
	@echo "OpenGL examples:"
	@echo "  ./bin/opengl_cube    (requires: brew install glfw glew)"
	@echo "  ./bin/opengl_teapot  (rotating teapot with texture cycling)"
	@echo ""
	@echo "Interpreter-only examples (run with: ./bin/nano examples/name.nano):"
	@echo "  $(INTERPRETER_EXAMPLES)"
	@echo ""

# SDL examples only
sdl: $(SDL_EXAMPLES)

# OpenGL examples only
opengl: $(OPENGL_EXAMPLES)

# Ensure compiler and interpreter exist by always checking parent Makefile
$(COMPILER):
	@cd .. && $(MAKE) bin/nanoc

$(INTERPRETER):
	@cd .. && $(MAKE) bin/nano

# === Automatic Module Building ===
# With the new module system, just invoke nanoc and it:
# 1. Detects imported modules
# 2. Reads module.json for C sources
# 3. Compiles C sources with pkg-config flags
# 4. Caches builds in .build/ directories
# 5. Links everything automatically!

# Individual target aliases
checkers-sdl: $(BIN_DIR)/checkers_sdl
boids-sdl: $(BIN_DIR)/boids_sdl
particles-sdl: $(BIN_DIR)/particles_sdl
falling-sand-sdl: $(BIN_DIR)/falling_sand_sdl

terrain-explorer-sdl: $(BIN_DIR)/terrain_explorer_sdl

opengl-cube: $(BIN_DIR)/opengl_cube
opengl-teapot: $(BIN_DIR)/opengl_teapot

# === Binary Build Rules ===

$(BIN_DIR)/checkers_sdl: checkers.nano $(COMPILER)
	@echo "Building Checkers SDL (with automatic module building)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/checkers.nano -o bin/checkers_sdl
	@echo "✓ Checkers SDL built: $@"

$(BIN_DIR)/boids_sdl: boids_sdl.nano $(COMPILER)
	@echo "Building Boids SDL (with automatic module building)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/boids_sdl.nano -o bin/boids_sdl
	@echo "✓ Boids SDL built: $@"

$(BIN_DIR)/particles_sdl: particles_sdl.nano $(COMPILER)
	@echo "Building Particles SDL (with automatic module building)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/particles_sdl.nano -o bin/particles_sdl
	@echo "✓ Particles SDL built: $@"

$(BIN_DIR)/falling_sand_sdl: falling_sand_sdl.nano $(COMPILER)
	@echo "Building Falling Sand SDL (with automatic module building)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/falling_sand_sdl.nano -o bin/falling_sand_sdl
	@echo "✓ Falling Sand SDL built: $@"

$(BIN_DIR)/terrain_explorer_sdl: terrain_explorer_sdl.nano $(COMPILER)
	@echo "Building Terrain Explorer SDL (with automatic module building + struct arrays)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/terrain_explorer_sdl.nano -o bin/terrain_explorer_sdl
	@echo "✓ Terrain Explorer SDL built: $@"

$(BIN_DIR)/opengl_cube: opengl_cube.nano $(COMPILER)
	@echo "Building OpenGL Rotating Cube (GLFW + GLEW)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/opengl_cube.nano -o bin/opengl_cube
	@echo "✓ OpenGL Cube built: $@"

$(BIN_DIR)/opengl_teapot: opengl_teapot/teapot.nano $(COMPILER)
	@echo "Building OpenGL Textured Teapot (GLFW + GLEW + procedural textures)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/opengl_teapot/teapot.nano -o bin/opengl_teapot
	@echo "✓ OpenGL Teapot built: $@"

# === Interpreter-Only Examples ===

interpreter:
	@echo "Running interpreter-only examples..."
	@echo ""
	@echo "=== Conway's Game of Life ==="
	@$(INTERPRETER) life_sdl.nano
	@echo ""
	@echo "=== Snake with AI ==="
	@$(INTERPRETER) snake_sdl.nano
	@echo ""
	@echo "=== Maze Generator ==="
	@$(INTERPRETER) maze_sdl.nano
	@echo ""
	@echo "=== Terrain Explorer (uses struct arrays - not yet supported in transpiler) ==="
	@$(INTERPRETER) terrain_explorer_sdl.nano
	@echo ""
	@echo "✓ All interpreter examples completed!"

# Run specific interpreter example
run-%: %.nano $(INTERPRETER)
	@echo "Running $< with interpreter..."
	@$(INTERPRETER) $<

# === Clean ===

clean:
	@echo "Cleaning example binaries and build artifacts..."
	rm -f $(GRAPHICS_EXAMPLES)
	rm -rf $(BIN_DIR)/*.dSYM
	rm -f $(BIN_DIR)/*.c
	rm -f ../obj/sdl_helpers_*.o
	rm -rf ../modules/*/.build/
	@echo "✓ Example binaries and artifacts cleaned"

# === Help ===

help:
	@echo "Nanolang Examples Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build all compiled examples (default)"
	@echo "  make sdl          - Build SDL examples only"
	@echo "  make interpreter  - Run all interpreter-only examples"
	@echo "  make checkers-sdl - Build checkers"
	@echo "  make run-life_sdl - Run life_sdl with interpreter"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Compiled examples (transpiler):"
	@echo "  checkers_sdl      - Full checkers game with AI"
	@echo "  boids_sdl         - Flocking simulation"
	@echo "  particles_sdl     - Particle explosion"
	@echo "  falling_sand_sdl  - Cellular automata sandbox"
	@echo ""
	@echo "Interpreter-only examples:"
	@echo "  life_sdl          - Conway's Game of Life"
	@echo "  snake_sdl         - Snake game with AI"
	@echo "  maze_sdl          - Maze generation"
	@echo "  terrain_explorer_sdl - Procedural terrain (uses struct arrays)"
	@echo ""
	@echo "Note: Automatic module building handles all C compilation!"
	@echo "      Just 'make' and everything is built automatically."
