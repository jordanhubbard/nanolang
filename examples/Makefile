# examples/Makefile - Build all nanolang game examples
#
# Usage:
#   make              - Build all examples
#   make checkers     - Build checkers game
#   make snake        - Build snake game
#   make boids        - Build boids simulation
#   make life         - Build Conway's Game of Life
#   make maze         - Build maze generator
#   make particles    - Build particle explosion
#   make clean        - Remove build artifacts

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g
BIN_DIR = ../bin
OBJ_DIR = ../obj
COMPILER = $(BIN_DIR)/nanoc
INTERPRETER = $(BIN_DIR)/nano

# SDL2 configuration
SDL2_CFLAGS := -I/opt/homebrew/include/SDL2 -I/usr/local/include/SDL2 -I/usr/include/SDL2
SDL2_LDFLAGS := -L/opt/homebrew/lib -L/usr/local/lib -L/usr/lib -lSDL2

# Check if SDL2_ttf is available
SDL2_TTF_AVAILABLE := $(shell pkg-config --exists sdl2_ttf 2>/dev/null && echo "yes" || echo "no")
ifeq ($(SDL2_TTF_AVAILABLE),yes)
    SDL2_CFLAGS += -DHAVE_SDL_TTF $(shell pkg-config --cflags sdl2_ttf)
    SDL2_LDFLAGS += $(shell pkg-config --libs sdl2_ttf)
else
    # Try common locations
    ifneq ($(wildcard /opt/homebrew/include/SDL2/SDL_ttf.h),)
        SDL2_CFLAGS += -DHAVE_SDL_TTF -I/opt/homebrew/include/SDL2
        SDL2_LDFLAGS += -L/opt/homebrew/lib -lSDL2_ttf
    else ifneq ($(wildcard /usr/local/include/SDL2/SDL_ttf.h),)
        SDL2_CFLAGS += -DHAVE_SDL_TTF -I/usr/local/include/SDL2
        SDL2_LDFLAGS += -L/usr/local/lib -lSDL2_ttf
    endif
endif

# Example binaries
CHECKERS = $(BIN_DIR)/checkers
BOIDS_SDL = $(BIN_DIR)/boids_sdl
PARTICLES_SDL = $(BIN_DIR)/particles_sdl
SNAKE = $(BIN_DIR)/snake
LIFE = $(BIN_DIR)/game_of_life
MAZE = $(BIN_DIR)/maze
BOIDS = $(BIN_DIR)/boids_complete

# SDL examples (require SDL2)
SDL_EXAMPLES = $(CHECKERS) $(BOIDS_SDL) $(PARTICLES_SDL)

# Interpreter-only examples (no compilation, just run with ./bin/nano)
INTERPRETER_EXAMPLES = $(SNAKE) $(LIFE) $(MAZE) $(BOIDS)

# All examples
ALL_EXAMPLES = $(SDL_EXAMPLES) $(INTERPRETER_EXAMPLES)

.PHONY: all sdl interpreter checkers boids-sdl particles-sdl snake life maze boids clean help

all: sdl interpreter
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "✅ All examples built successfully!"
	@echo "═══════════════════════════════════════════════════"
	@echo ""
	@echo "SDL Games (compiled):"
	@echo "  $(CHECKERS)      - Full checkers game with AI"
	@echo "  $(BOIDS_SDL)     - Visual flocking simulation"
	@echo "  $(PARTICLES_SDL) - Particle physics explosion"
	@echo ""
	@echo "Interpreter Games (run with ./bin/nano):"
	@echo "  ./bin/nano examples/snake.nano         - Snake with AI"
	@echo "  ./bin/nano examples/game_of_life.nano  - Conway's Life"
	@echo "  ./bin/nano examples/maze.nano          - Maze generator"
	@echo "  ./bin/nano examples/boids_complete.nano - Text boids"
	@echo ""

sdl: $(SDL_EXAMPLES)
	@echo "✓ SDL examples built"

interpreter: $(INTERPRETER_EXAMPLES)
	@echo "✓ Interpreter examples verified"

# Checkers - Full-featured SDL game with AI, king pieces, jump logic
$(CHECKERS): checkers.nano ../modules/sdl/sdl.nano ../modules/sdl_helpers/sdl_helpers.nano ../modules/sdl_helpers/sdl_helpers.c $(COMPILER) | $(BIN_DIR) $(OBJ_DIR)
	@echo "Building checkers game..."
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) -c ../modules/sdl_helpers/sdl_helpers.c -o $(OBJ_DIR)/sdl_helpers_checkers.o
	NANO_MODULE_PATH=../modules $(COMPILER) checkers.nano -o $(CHECKERS) --keep-c
	@echo "#include <stdint.h>" > $(BIN_DIR)/checkers_externs.c
	@echo "extern int64_t nl_sdl_render_fill_rect(int64_t renderer, int64_t x, int64_t y, int64_t w, int64_t h);" >> $(BIN_DIR)/checkers_externs.c
	@echo "extern int64_t nl_sdl_poll_event_quit(void);" >> $(BIN_DIR)/checkers_externs.c
	@echo "extern int64_t nl_sdl_poll_mouse_click(void);" >> $(BIN_DIR)/checkers_externs.c
	@echo "extern int64_t nl_sdl_render_text_solid(int64_t renderer, int64_t font, const char* text, int64_t x, int64_t y, int64_t r, int64_t g, int64_t b, int64_t a);" >> $(BIN_DIR)/checkers_externs.c
	@echo "extern int64_t nl_sdl_render_text_blended(int64_t renderer, int64_t font, const char* text, int64_t x, int64_t y, int64_t r, int64_t g, int64_t b, int64_t a);" >> $(BIN_DIR)/checkers_externs.c
	@cat $(BIN_DIR)/checkers_externs.c $(BIN_DIR)/checkers.c > $(BIN_DIR)/checkers_tmp.c && mv $(BIN_DIR)/checkers_tmp.c $(BIN_DIR)/checkers.c
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) $(BIN_DIR)/checkers.c $(OBJ_DIR)/sdl_helpers_checkers.o -o $(CHECKERS) $(SDL2_LDFLAGS)
	@rm -f $(BIN_DIR)/checkers_externs.c
	@echo "✓ Checkers built: $(CHECKERS)"

# Boids SDL - Visual flocking simulation
$(BOIDS_SDL): boids_sdl.nano ../modules/sdl/sdl.nano ../modules/sdl_helpers/sdl_helpers.nano ../modules/sdl_helpers/sdl_helpers.c $(COMPILER) | $(BIN_DIR) $(OBJ_DIR)
	@echo "Building Boids SDL simulation..."
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) -c ../modules/sdl_helpers/sdl_helpers.c -o $(OBJ_DIR)/sdl_helpers_boids.o
	NANO_MODULE_PATH=../modules $(COMPILER) boids_sdl.nano -o $(BOIDS_SDL) --keep-c
	@echo "#include <stdint.h>" > $(BIN_DIR)/boids_sdl_externs.c
	@echo "extern int64_t nl_sdl_render_fill_rect(int64_t renderer, int64_t x, int64_t y, int64_t w, int64_t h);" >> $(BIN_DIR)/boids_sdl_externs.c
	@cat $(BIN_DIR)/boids_sdl_externs.c $(BIN_DIR)/boids_sdl.c > $(BIN_DIR)/boids_sdl_tmp.c && mv $(BIN_DIR)/boids_sdl_tmp.c $(BIN_DIR)/boids_sdl.c
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) $(BIN_DIR)/boids_sdl.c $(OBJ_DIR)/sdl_helpers_boids.o -o $(BOIDS_SDL) $(SDL2_LDFLAGS)
	@rm -f $(BIN_DIR)/boids_sdl_externs.c
	@echo "✓ Boids SDL built: $(BOIDS_SDL)"

# Particles SDL - Particle explosion with physics
$(PARTICLES_SDL): particles_sdl.nano ../modules/sdl/sdl.nano ../modules/sdl_helpers/sdl_helpers.nano ../modules/sdl_helpers/sdl_helpers.c $(COMPILER) | $(BIN_DIR) $(OBJ_DIR)
	@echo "Building Particles SDL demo..."
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) -c ../modules/sdl_helpers/sdl_helpers.c -o $(OBJ_DIR)/sdl_helpers_particles.o
	NANO_MODULE_PATH=../modules $(COMPILER) particles_sdl.nano -o $(PARTICLES_SDL) --keep-c
	@echo "#include <stdint.h>" > $(BIN_DIR)/particles_sdl_externs.c
	@echo "extern int64_t nl_sdl_render_fill_rect(int64_t renderer, int64_t x, int64_t y, int64_t w, int64_t h);" >> $(BIN_DIR)/particles_sdl_externs.c
	@cat $(BIN_DIR)/particles_sdl_externs.c $(BIN_DIR)/particles_sdl.c > $(BIN_DIR)/particles_sdl_tmp.c && mv $(BIN_DIR)/particles_sdl_tmp.c $(BIN_DIR)/particles_sdl.c
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) $(BIN_DIR)/particles_sdl.c $(OBJ_DIR)/sdl_helpers_particles.o -o $(PARTICLES_SDL) $(SDL2_LDFLAGS)
	@rm -f $(BIN_DIR)/particles_sdl_externs.c
	@echo "✓ Particles SDL built: $(PARTICLES_SDL)"

# Interpreter-only examples - just verify they exist and compile shadow tests
$(SNAKE): snake.nano $(COMPILER) $(INTERPRETER)
	@echo "Verifying Snake game..."
	@$(COMPILER) snake.nano -o $(SNAKE) > /dev/null 2>&1
	@echo "✓ Snake verified (run with: ./bin/nano examples/snake.nano)"

$(LIFE): game_of_life.nano $(COMPILER) $(INTERPRETER)
	@echo "Verifying Conway's Game of Life..."
	@$(COMPILER) game_of_life.nano -o $(LIFE) > /dev/null 2>&1
	@echo "✓ Game of Life verified (run with: ./bin/nano examples/game_of_life.nano)"

$(MAZE): maze.nano $(COMPILER) $(INTERPRETER)
	@echo "Verifying Maze generator..."
	@$(COMPILER) maze.nano -o $(MAZE) > /dev/null 2>&1
	@echo "✓ Maze verified (run with: ./bin/nano examples/maze.nano)"

$(BOIDS): boids_complete.nano $(COMPILER) $(INTERPRETER)
	@echo "Verifying Boids simulation..."
	@$(COMPILER) boids_complete.nano -o $(BOIDS) > /dev/null 2>&1
	@echo "✓ Boids verified (run with: ./bin/nano examples/boids_complete.nano)"

# Convenience targets
checkers: $(CHECKERS)

boids-sdl: $(BOIDS_SDL)

particles-sdl: $(PARTICLES_SDL)

snake: $(SNAKE)

life: $(LIFE)

maze: $(MAZE)

boids: $(BOIDS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	@echo "Cleaning example binaries..."
	rm -f $(ALL_EXAMPLES)
	rm -f $(BIN_DIR)/checkers.c $(BIN_DIR)/boids_sdl.c $(BIN_DIR)/particles_sdl.c
	rm -f $(OBJ_DIR)/sdl_helpers_*.o
	@echo "✓ Examples cleaned"

help:
	@echo "Examples Makefile targets:"
	@echo "  make              - Build all examples"
	@echo "  make sdl          - Build SDL examples only"
	@echo "  make interpreter  - Verify interpreter examples"
	@echo ""
	@echo "Individual examples:"
	@echo "  make checkers     - Build checkers game"
	@echo "  make boids-sdl    - Build visual boids simulation"
	@echo "  make particles-sdl - Build particle explosion"
	@echo "  make snake        - Verify snake game"
	@echo "  make life         - Verify Conway's Life"
	@echo "  make maze         - Verify maze generator"
	@echo "  make boids        - Verify text boids"
	@echo ""
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help"

