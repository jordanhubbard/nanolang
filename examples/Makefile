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
	$(BIN_DIR)/falling_sand \
	$(BIN_DIR)/mod_player \
	$(BIN_DIR)/raytracer_simple \
	$(BIN_DIR)/asteroids_sdl \
	$(BIN_DIR)/terrain_explorer_sdl

# Additional compiled examples (non-graphics)
COMPILED_EXAMPLES = \
	$(BIN_DIR)/snake \
	$(BIN_DIR)/game_of_life \
	$(BIN_DIR)/random_simple

# Note: array<struct> is now fully supported! (fixed in commit 4952d65)
# terrain_explorer_sdl uses array<Tile> and compiles successfully.

# OpenGL examples (GLFW + GLEW)
OPENGL_EXAMPLES = \
	$(BIN_DIR)/opengl_cube \
	$(BIN_DIR)/opengl_teapot

# All graphics examples
GRAPHICS_EXAMPLES = $(SDL_EXAMPLES) $(OPENGL_EXAMPLES)

# All compiled examples
ALL_COMPILED = $(GRAPHICS_EXAMPLES) $(COMPILED_EXAMPLES)

# === Interpreter Demo Examples ===
# Simple examples perfect for demonstrating the interpreter
# Run with: ../bin/nano examples/<filename>
# Note: Some of these can also be compiled, but are great for interpreter demos

INTERPRETER_DEMO_EXAMPLES = \
	hello.nano \
	calculator.nano \
	factorial.nano \
	fibonacci.nano \
	primes.nano \
	game_of_life.nano \
	snake.nano \
	maze.nano

# Interpreter-only examples (use features transpiler doesn't support yet)
INTERPRETER_ONLY_EXAMPLES = \
	game_of_life.nano \
	snake.nano \
	maze.nano

# === Targets ===

.PHONY: all sdl opengl interpreter clean help checkers-sdl boids-sdl particles-sdl falling-sand-sdl terrain-explorer-sdl opengl-cube opengl-teapot run-% force-rebuild

# Default: build all compiled examples
all: $(COMPILER) $(ALL_COMPILED)
	@echo ""
	@echo "✓ All compiled examples built successfully!"
	@echo ""
	@echo "SDL examples:"
	@echo "  ./bin/checkers_sdl           - Full checkers game with AI"
	@echo "  ./bin/boids_sdl              - Flocking simulation (Craig Reynolds' Boids)"
	@echo "  ./bin/particles_sdl          - Particle explosion effects (nested arrays!)"
	@echo "  ./bin/falling_sand           - Cellular automata sandbox"
	@echo "  ./bin/mod_player             - MOD/XM music player (requires music files)"
	@echo "  ./bin/raytracer_simple       - Real-time ray tracer with mouse-controlled light"
	@echo "  ./bin/asteroids_sdl          - Classic arcade game (parallel arrays pattern)"
	@echo "  ./bin/terrain_explorer_sdl   - 3D terrain with mouse camera (array<struct>!)"
	@echo ""
	@echo "Compiled terminal examples:"
	@echo "  ./bin/snake                  - Classic snake game"
	@echo "  ./bin/game_of_life           - Conway's Game of Life"
	@echo "  ./bin/random_simple          - Random number generation"
	@echo ""
	@echo "OpenGL examples (requires: brew install glfw glew):"
	@echo "  ./bin/opengl_cube          - Rotating textured cube"
	@echo "  ./bin/opengl_teapot        - Rotating teapot with texture cycling"
	@echo ""
	@echo "Interpreter demo examples (run with: ./bin/nano examples/<filename>):"
	@echo "  hello.nano"
	@echo "  calculator.nano"
	@echo "  factorial.nano"
	@echo "  fibonacci.nano"
	@echo "  primes.nano"
	@echo "  game_of_life.nano"
	@echo "  snake.nano"
	@echo "  maze.nano"
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
raytracer-simple: $(BIN_DIR)/raytracer_simple
asteroids-sdl: $(BIN_DIR)/asteroids_sdl
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
	@echo "Building Particles SDL (particle explosion effects)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/particles_sdl.nano -o bin/particles_sdl
	@echo "✓ Particles SDL built: $@"

$(BIN_DIR)/falling_sand: falling_sand.nano $(COMPILER)
	@echo "Building Falling Sand (cellular automata sandbox)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/falling_sand.nano -o bin/falling_sand
	@echo "✓ Falling Sand built: $@"

$(BIN_DIR)/terrain_explorer_sdl: terrain_explorer_sdl.nano $(COMPILER)
	@echo "Building Terrain Explorer SDL (3D terrain with mouse camera)..."
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

$(BIN_DIR)/raytracer_simple: raytracer_simple.nano $(COMPILER)
	@echo "Building Ray Tracer (real-time with mouse-controlled light)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/raytracer_simple.nano -o bin/raytracer_simple
	@echo "✓ Ray Tracer built: $@"

$(BIN_DIR)/mod_player: mod_player.nano $(COMPILER)
	@echo "Building MOD Player (SDL_mixer music player)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/mod_player.nano -o bin/mod_player
	@echo "✓ MOD Player built: $@"

$(BIN_DIR)/asteroids_sdl: asteroids_sdl.nano $(COMPILER)
	@echo "Building Asteroids (classic arcade game with parallel arrays)..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/asteroids_sdl.nano -o bin/asteroids_sdl
	@echo "✓ Asteroids built: $@"

# === Non-Graphics Compiled Examples ===

$(BIN_DIR)/snake: snake.nano $(COMPILER)
	@echo "Building Snake (terminal game)..."
	cd .. && ./bin/nanoc examples/snake.nano -o bin/snake
	@echo "✓ Snake built: $@"

$(BIN_DIR)/game_of_life: game_of_life.nano $(COMPILER)
	@echo "Building Game of Life (Conway's cellular automaton)..."
	cd .. && ./bin/nanoc examples/game_of_life.nano -o bin/game_of_life
	@echo "✓ Game of Life built: $@"

$(BIN_DIR)/random_simple: 24_random_simple.nano $(COMPILER)
	@echo "Building Random Simple (number generation demo)..."
	cd .. && ./bin/nanoc examples/24_random_simple.nano -o bin/random_simple
	@echo "✓ Random Simple built: $@"

# === Interpreter Demo Examples ===

interpreter:
	@echo "Running interpreter demo examples..."
	@echo ""
	@echo "=== Hello World ==="
	@$(INTERPRETER) hello.nano
	@echo ""
	@echo "=== Calculator ==="
	@$(INTERPRETER) calculator.nano
	@echo ""
	@echo "=== Factorial ==="
	@$(INTERPRETER) factorial.nano
	@echo ""
	@echo "=== Fibonacci ==="
	@$(INTERPRETER) fibonacci.nano
	@echo ""
	@echo "=== Primes ==="
	@$(INTERPRETER) primes.nano
	@echo ""
	@echo "=== Conway's Game of Life ==="
	@$(INTERPRETER) game_of_life.nano
	@echo ""
	@echo "=== Snake with AI ==="
	@$(INTERPRETER) snake.nano
	@echo ""
	@echo "=== Maze Generator ==="
	@$(INTERPRETER) maze.nano
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
	@echo "  make interpreter  - Run all interpreter demo examples"
	@echo "  make checkers-sdl - Build checkers"
	@echo "  make run-hello        - Run hello with interpreter"
	@echo "  make run-calculator   - Run calculator with interpreter"
	@echo "  make run-factorial    - Run factorial with interpreter"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Compiled examples (transpiler):"
	@echo "  checkers-sdl         - Full checkers game with AI"
	@echo "  boids-sdl            - Flocking simulation"
	@echo "  particles-sdl        - Particle explosion"
	@echo "  falling-sand-sdl     - Cellular automata sandbox"
	@echo "  terrain-explorer-sdl - 3D terrain with mouse camera"
	@echo "  raytracer-simple     - Real-time ray tracer"
	@echo "  opengl-cube          - Rotating OpenGL cube"
	@echo "  opengl-teapot        - Rotating OpenGL teapot"
	@echo ""
	@echo "Interpreter demo examples:"
	@echo "  hello             - Hello World (simplest example)"
	@echo "  calculator        - Basic arithmetic operations"
	@echo "  factorial         - Recursive factorial calculation"
	@echo "  fibonacci         - Fibonacci sequence"
	@echo "  primes            - Prime number checker"
	@echo "  game_of_life      - Conway's Game of Life"
	@echo "  snake             - Snake game with AI"
	@echo "  maze              - Maze generation"
	@echo ""
	@echo "Note: Automatic module building handles all C compilation!"
	@echo "      Just 'make' and everything is built automatically."
