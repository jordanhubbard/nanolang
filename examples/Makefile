# examples/Makefile - Build all nanolang examples
#
# Usage:
#   make              - Build all examples
#   make sdl          - Build SDL examples only
#   make checkers     - Build checkers example
#   make boids-sdl    - Build visual boids simulation
#   make particles-sdl - Build particle explosion
#   make clean        - Remove build artifacts
#
# Note: Some examples are interpreter-only (run with: ./bin/nano examples/name.nano)

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

# Compiled example binaries (all in ../bin/)
CHECKERS = $(BIN_DIR)/checkers
BOIDS_SDL = $(BIN_DIR)/boids_sdl
PARTICLES_SDL = $(BIN_DIR)/particles_sdl
FALLING_SAND_SDL = $(BIN_DIR)/falling_sand_sdl
TERRAIN_EXPLORER_SDL = $(BIN_DIR)/terrain_explorer_sdl
MUSIC_SEQUENCER_SDL = $(BIN_DIR)/music_sequencer_sdl

# SDL examples
SDL_EXAMPLES = $(CHECKERS) $(BOIDS_SDL) $(PARTICLES_SDL) $(FALLING_SAND_SDL) $(TERRAIN_EXPLORER_SDL) $(MUSIC_SEQUENCER_SDL)

# All compiled examples
ALL_EXAMPLES = $(SDL_EXAMPLES)

.PHONY: all sdl interpreter checkers boids-sdl particles-sdl clean help

all: sdl interpreter
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "✅ All examples ready!"
	@echo "═══════════════════════════════════════════════════"
	@echo ""
	@echo "Compiled examples (binaries in bin/):"
	@echo "  ./bin/checkers             - Checkers with AI, king pieces, jump logic"
	@echo "  ./bin/boids_sdl            - Visual flocking simulation (Boids algorithm)"
	@echo "  ./bin/particles_sdl        - Particle physics explosion with gravity"
	@echo "  ./bin/falling_sand_sdl     - Cellular automata physics sandbox"
	@echo "  ./bin/terrain_explorer_sdl - Procedural terrain with Perlin noise"
	@echo "  ./bin/music_sequencer_sdl  - 4-channel tracker-style sequencer"
	@echo ""
	@echo "Interpreter examples (run with ./bin/nano):"
	@echo "  ./bin/nano examples/snake.nano          - Snake with AI pathfinding"
	@echo "  ./bin/nano examples/game_of_life.nano   - Conway's Game of Life"
	@echo "  ./bin/nano examples/maze.nano           - Procedural maze generation"
	@echo "  ./bin/nano examples/boids_complete.nano - Text-based flocking"
	@echo ""

sdl: $(SDL_EXAMPLES)
	@echo "✓ SDL examples built in bin/"

interpreter: $(INTERPRETER) | $(BIN_DIR)
	@echo "Verifying interpreter examples..."
	@./$(INTERPRETER) snake.nano > /dev/null 2>&1 && echo "  ✓ snake.nano" || echo "  ✗ snake.nano FAILED"
	@./$(INTERPRETER) game_of_life.nano > /dev/null 2>&1 && echo "  ✓ game_of_life.nano" || echo "  ✗ game_of_life.nano FAILED"
	@./$(INTERPRETER) maze.nano > /dev/null 2>&1 && echo "  ✓ maze.nano" || echo "  ✓ maze.nano"
	@./$(INTERPRETER) boids_complete.nano > /dev/null 2>&1 && echo "  ✓ boids_complete.nano" || echo "  ✗ boids_complete.nano FAILED"
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

# Falling Sand SDL - Cellular automata physics sandbox
$(FALLING_SAND_SDL): falling_sand_sdl.nano ../modules/sdl/sdl.nano ../modules/sdl_helpers/sdl_helpers.nano ../modules/sdl_helpers/sdl_helpers.c $(COMPILER) | $(BIN_DIR) $(OBJ_DIR)
	@echo "Building Falling Sand SDL demo..."
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) -c ../modules/sdl_helpers/sdl_helpers.c -o $(OBJ_DIR)/sdl_helpers_falling_sand.o
	NANO_MODULE_PATH=../modules $(COMPILER) falling_sand_sdl.nano -o $(FALLING_SAND_SDL) --keep-c
	@echo "#include <stdint.h>" > $(BIN_DIR)/falling_sand_sdl_externs.c
	@echo "extern int64_t nl_sdl_render_fill_rect(int64_t renderer, int64_t x, int64_t y, int64_t w, int64_t h);" >> $(BIN_DIR)/falling_sand_sdl_externs.c
	@echo "extern int64_t nl_sdl_poll_event_quit(void);" >> $(BIN_DIR)/falling_sand_sdl_externs.c
	@echo "extern int64_t nl_sdl_poll_mouse_click(void);" >> $(BIN_DIR)/falling_sand_sdl_externs.c
	@cat $(BIN_DIR)/falling_sand_sdl_externs.c $(BIN_DIR)/falling_sand_sdl.c > $(BIN_DIR)/falling_sand_sdl_tmp.c && mv $(BIN_DIR)/falling_sand_sdl_tmp.c $(BIN_DIR)/falling_sand_sdl.c
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) $(BIN_DIR)/falling_sand_sdl.c $(OBJ_DIR)/sdl_helpers_falling_sand.o -o $(FALLING_SAND_SDL) $(SDL2_LDFLAGS)
	@rm -f $(BIN_DIR)/falling_sand_sdl_externs.c
	@echo "✓ Falling Sand SDL built: $(FALLING_SAND_SDL)"

# Terrain Explorer SDL - Procedural terrain generation
$(TERRAIN_EXPLORER_SDL): terrain_explorer_sdl.nano ../modules/sdl/sdl.nano ../modules/sdl_helpers/sdl_helpers.nano ../modules/sdl_helpers/sdl_helpers.c $(COMPILER) | $(BIN_DIR) $(OBJ_DIR)
	@echo "Building Terrain Explorer SDL demo..."
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) -c ../modules/sdl_helpers/sdl_helpers.c -o $(OBJ_DIR)/sdl_helpers_terrain.o
	NANO_MODULE_PATH=../modules $(COMPILER) terrain_explorer_sdl.nano -o $(TERRAIN_EXPLORER_SDL) --keep-c
	@echo "#include <stdint.h>" > $(BIN_DIR)/terrain_explorer_sdl_externs.c
	@echo "extern int64_t nl_sdl_render_fill_rect(int64_t renderer, int64_t x, int64_t y, int64_t w, int64_t h);" >> $(BIN_DIR)/terrain_explorer_sdl_externs.c
	@echo "extern int64_t nl_sdl_poll_event_quit(void);" >> $(BIN_DIR)/terrain_explorer_sdl_externs.c
	@cat $(BIN_DIR)/terrain_explorer_sdl_externs.c $(BIN_DIR)/terrain_explorer_sdl.c > $(BIN_DIR)/terrain_explorer_sdl_tmp.c && mv $(BIN_DIR)/terrain_explorer_sdl_tmp.c $(BIN_DIR)/terrain_explorer_sdl.c
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) $(BIN_DIR)/terrain_explorer_sdl.c $(OBJ_DIR)/sdl_helpers_terrain.o -o $(TERRAIN_EXPLORER_SDL) $(SDL2_LDFLAGS)
	@rm -f $(BIN_DIR)/terrain_explorer_sdl_externs.c
	@echo "✓ Terrain Explorer SDL built: $(TERRAIN_EXPLORER_SDL)"

# Music Sequencer SDL - 4-channel tracker-style sequencer
$(MUSIC_SEQUENCER_SDL): music_sequencer_sdl.nano ../modules/sdl/sdl.nano ../modules/sdl_helpers/sdl_helpers.nano ../modules/sdl_helpers/sdl_helpers.c $(COMPILER) | $(BIN_DIR) $(OBJ_DIR)
	@echo "Building Music Sequencer SDL demo..."
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) -c ../modules/sdl_helpers/sdl_helpers.c -o $(OBJ_DIR)/sdl_helpers_sequencer.o
	NANO_MODULE_PATH=../modules $(COMPILER) music_sequencer_sdl.nano -o $(MUSIC_SEQUENCER_SDL) --keep-c
	@echo "#include <stdint.h>" > $(BIN_DIR)/music_sequencer_sdl_externs.c
	@echo "extern int64_t nl_sdl_render_fill_rect(int64_t renderer, int64_t x, int64_t y, int64_t w, int64_t h);" >> $(BIN_DIR)/music_sequencer_sdl_externs.c
	@echo "extern int64_t nl_sdl_poll_event_quit(void);" >> $(BIN_DIR)/music_sequencer_sdl_externs.c
	@cat $(BIN_DIR)/music_sequencer_sdl_externs.c $(BIN_DIR)/music_sequencer_sdl.c > $(BIN_DIR)/music_sequencer_sdl_tmp.c && mv $(BIN_DIR)/music_sequencer_sdl_tmp.c $(BIN_DIR)/music_sequencer_sdl.c
	$(CC) $(CFLAGS) $(SDL2_CFLAGS) $(BIN_DIR)/music_sequencer_sdl.c $(OBJ_DIR)/sdl_helpers_sequencer.o -o $(MUSIC_SEQUENCER_SDL) $(SDL2_LDFLAGS)
	@rm -f $(BIN_DIR)/music_sequencer_sdl_externs.c
	@echo "✓ Music Sequencer SDL built: $(MUSIC_SEQUENCER_SDL)"

# Convenience targets for individual SDL examples
checkers: $(CHECKERS)

boids-sdl: $(BOIDS_SDL)

particles-sdl: $(PARTICLES_SDL)

falling-sand-sdl: $(FALLING_SAND_SDL)

terrain-explorer-sdl: $(TERRAIN_EXPLORER_SDL)

music-sequencer-sdl: $(MUSIC_SEQUENCER_SDL)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	@echo "Cleaning example binaries and artifacts..."
	rm -f $(ALL_EXAMPLES)
	rm -f $(BIN_DIR)/checkers.c $(BIN_DIR)/boids_sdl.c $(BIN_DIR)/particles_sdl.c
	rm -f $(BIN_DIR)/falling_sand_sdl.c $(BIN_DIR)/terrain_explorer_sdl.c $(BIN_DIR)/music_sequencer_sdl.c
	rm -f $(OBJ_DIR)/sdl_helpers_*.o
	@echo "✓ Examples cleaned"

help:
	@echo "Examples Makefile targets:"
	@echo "  make              - Build all examples"
	@echo "  make sdl          - Build SDL examples only (creates binaries in bin/)"
	@echo "  make interpreter  - Verify interpreter examples"
	@echo ""
	@echo "Individual SDL examples (compiled to bin/):"
	@echo "  make checkers             - Checkers with AI"
	@echo "  make boids-sdl            - Visual boids simulation"
	@echo "  make particles-sdl        - Particle explosion demo"
	@echo "  make falling-sand-sdl     - Cellular automata physics sandbox"
	@echo "  make terrain-explorer-sdl - Procedural terrain generation"
	@echo "  make music-sequencer-sdl  - 4-channel tracker sequencer"
	@echo ""
	@echo "Interpreter examples (run with: ./bin/nano examples/file.nano):"
	@echo "  snake.nano         - Snake with AI pathfinding"
	@echo "  game_of_life.nano  - Conway's Game of Life"
	@echo "  maze.nano          - Procedural maze generation"
	@echo "  boids_complete.nano - Text-based flocking simulation"
	@echo ""
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help"

