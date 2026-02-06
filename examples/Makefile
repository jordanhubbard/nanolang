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

NL_SOURCES := $(filter-out language/nl_pi_calculator.nano language/nl_pi_chudnovsky.nano,$(wildcard language/nl_*.nano))
NL_EXAMPLES := $(patsubst language/%.nano,$(BIN_DIR)/%,$(NL_SOURCES))
NL_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(NL_EXAMPLES)))

# === REPL Examples ===

# Most REPL examples require the readline module. Skip if it's not available.
READLINE_AVAILABLE := $(shell sh -c 'if command -v pkg-config >/dev/null 2>&1; then pkg-config --exists readline >/dev/null 2>&1 && echo yes; elif [ -x /opt/homebrew/bin/pkg-config ]; then /opt/homebrew/bin/pkg-config --exists readline >/dev/null 2>&1 && echo yes; fi')

ifeq ($(READLINE_AVAILABLE),yes)
REPL_SOURCES := $(filter-out language/readline_repl.nano language/full_repl.nano language/multi_type_repl.nano language/multiline_repl.nano,$(wildcard language/*_repl.nano))
else
REPL_SOURCES :=
endif
REPL_EXAMPLES := $(patsubst language/%.nano,$(BIN_DIR)/%,$(REPL_SOURCES))
REPL_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(REPL_EXAMPLES)))

# === Advanced Examples ===

ADVANCED_SOURCES := $(filter-out advanced/math_helper.nano,$(wildcard advanced/*.nano))
ADVANCED_EXAMPLES := $(patsubst advanced/%.nano,$(BIN_DIR)/%,$(ADVANCED_SOURCES))
ADVANCED_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(ADVANCED_EXAMPLES)))

# === Data Examples ===

DATA_SOURCES := $(wildcard data/*.nano)
DATA_EXAMPLES := $(patsubst data/%.nano,$(BIN_DIR)/%,$(DATA_SOURCES))
DATA_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(DATA_EXAMPLES)))

# === Debugging & Validation Examples ===

DEBUG_SOURCES := $(wildcard debug/*.nano)
DEBUG_EXAMPLES := $(patsubst debug/%.nano,$(BIN_DIR)/%,$(DEBUG_SOURCES))
DEBUG_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(DEBUG_EXAMPLES)))

# === Network Examples ===

NETWORK_SOURCES := $(wildcard network/*.nano)
NETWORK_EXAMPLES := $(patsubst network/%.nano,$(BIN_DIR)/%,$(NETWORK_SOURCES))
NETWORK_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(NETWORK_EXAMPLES)))

# === Physics Examples ===

BULLET_AVAILABLE := $(shell pkg-config --exists bullet >/dev/null 2>&1 && echo yes || echo no)
PHYSICS_SOURCES := $(wildcard physics/*.nano)
PHYSICS_EXAMPLES := $(patsubst physics/%.nano,$(BIN_DIR)/%,$(PHYSICS_SOURCES))
PHYSICS_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(PHYSICS_EXAMPLES)))

ifeq ($(BULLET_AVAILABLE),no)
PHYSICS_SOURCES :=
PHYSICS_EXAMPLES :=
PHYSICS_NAMES :=
endif

# === Playground Examples ===

PLAYGROUND_SOURCES := $(wildcard playground/*.nano)
PLAYGROUND_EXAMPLES := $(patsubst playground/%.nano,$(BIN_DIR)/%,$(PLAYGROUND_SOURCES)) $(BIN_DIR)/playground
PLAYGROUND_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(PLAYGROUND_EXAMPLES)))

# === Root Examples ===

ROOT_SOURCES := nano_tools_lint.nano sdl_example_launcher.nano
ROOT_EXAMPLES := $(patsubst %.nano,$(BIN_DIR)/%,$(ROOT_SOURCES))
ROOT_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(ROOT_EXAMPLES)))

# === OPL Examples ===

OPL_SOURCES := $(filter-out opl/opl_ast.nano opl/opl_codegen.nano opl/opl_compile.nano opl/opl_json.nano opl/opl_lexer.nano opl/opl_parser.nano opl/opl_validate.nano,$(wildcard opl/*.nano))
OPL_EXAMPLES := $(patsubst opl/%.nano,$(BIN_DIR)/%,$(OPL_SOURCES)) $(BIN_DIR)/opl
OPL_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(OPL_EXAMPLES)))

# === SDL Examples (sdl_*) ===
# Require SDL2 library

SDL_EXAMPLES = \
	$(patsubst graphics/%.nano,$(BIN_DIR)/%,$(wildcard graphics/sdl_*.nano)) \
	$(patsubst audio/%.nano,$(BIN_DIR)/%,$(wildcard audio/sdl_*.nano)) \
	$(patsubst games/%.nano,$(BIN_DIR)/%,$(wildcard games/sdl_*.nano)) \
	$(BIN_DIR)/sdl_example_launcher
SDL_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(SDL_EXAMPLES)))

# === NCurses Examples (ncurses_*) ===
# Require ncurses library

NCURSES_SOURCES := $(wildcard terminal/ncurses_*.nano)
NCURSES_EXAMPLES := $(patsubst terminal/%.nano,$(BIN_DIR)/%,$(NCURSES_SOURCES))
NCURSES_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(NCURSES_EXAMPLES)))

# === OpenGL Examples (opengl_*) ===
# Require GLFW + GLEW

OPENGL_SOURCES := $(wildcard opengl/*.nano)
OPENGL_EXAMPLES := $(patsubst opengl/%.nano,$(BIN_DIR)/%,$(OPENGL_SOURCES))
OPENGL_NAMES := $(sort $(patsubst $(BIN_DIR)/%,%,$(OPENGL_EXAMPLES)))

# === All Default Examples ===

ALL_EXAMPLES = \
	$(NL_EXAMPLES) \
	$(REPL_EXAMPLES) \
	$(ADVANCED_EXAMPLES) \
	$(DATA_EXAMPLES) \
	$(DEBUG_EXAMPLES) \
	$(NETWORK_EXAMPLES) \
	$(PHYSICS_EXAMPLES) \
	$(PLAYGROUND_EXAMPLES) \
	$(ROOT_EXAMPLES) \
	$(OPL_EXAMPLES) \
	$(SDL_EXAMPLES) \
	$(NCURSES_EXAMPLES) \
	$(OPENGL_EXAMPLES)

# === Targets ===

.PHONY: all build examples-available nl opl sdl ncurses opengl debug clean help launcher

# Default: build all examples
all: build

# Build all examples (STRICT: fails if module dependencies missing)
# Run 'make modules' first to see what's needed
build: $(COMPILER) $(ALL_EXAMPLES)
	@echo ""
	@echo "=========================================="
	@echo "✓ All Examples Built Successfully"
	@echo "=========================================="
	@echo ""
	@echo "Nanolang-only examples (nl_*):"
	@$(if $(strip $(NL_NAMES)),$(foreach ex,$(NL_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "REPL examples:"
	@$(if $(strip $(REPL_NAMES)),$(foreach ex,$(REPL_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "Advanced examples:"
	@$(if $(strip $(ADVANCED_NAMES)),$(foreach ex,$(ADVANCED_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "Data examples:"
	@$(if $(strip $(DATA_NAMES)),$(foreach ex,$(DATA_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "Debugging & Validation examples:"
	@$(if $(strip $(DEBUG_NAMES)),$(foreach ex,$(DEBUG_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "Network examples:"
	@$(if $(strip $(NETWORK_NAMES)),$(foreach ex,$(NETWORK_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "Physics examples:"
	@$(if $(strip $(PHYSICS_NAMES)),$(foreach ex,$(PHYSICS_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "Playground examples:"
	@$(if $(strip $(PLAYGROUND_NAMES)),$(foreach ex,$(PLAYGROUND_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "Tools & launchers:"
	@$(if $(strip $(ROOT_NAMES)),$(foreach ex,$(ROOT_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "OPL examples:"
	@$(if $(strip $(OPL_NAMES)),$(foreach ex,$(OPL_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "SDL examples (sdl_*):"
	@$(if $(strip $(SDL_NAMES)),$(foreach ex,$(SDL_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "NCurses examples (ncurses_*):"
	@$(if $(strip $(NCURSES_NAMES)),$(foreach ex,$(NCURSES_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "OpenGL examples (opengl_*):"
	@$(if $(strip $(OPENGL_NAMES)),$(foreach ex,$(OPENGL_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
	@echo "To launch the example browser:"
	@echo "  cd examples && make launcher"
	@echo "  # or from repo root: make launcher"
	@echo ""

# Build examples (GRACEFUL: skip examples with missing dependencies)
# Useful for quick iteration when you don't have all system libraries
examples-available: $(COMPILER)
	@echo "Building available examples (skipping those with missing dependencies)..."
	@echo ""
	@$(MAKE) -k $(ALL_EXAMPLES) 2>&1 | grep -v "Skipping module" || true
	@echo ""
	@echo "=========================================="
	@echo "Available Examples Built"
	@echo "=========================================="
	@echo ""
	@echo "Note: Examples requiring unavailable modules were skipped."
	@echo "      Run 'make modules' to see what's needed for all examples."
	@echo ""

# Example launcher (SDL app-store style)
launcher: $(COMPILER) $(BIN_DIR)/sdl_example_launcher
	@echo "Launching... (ESC or Quit to exit)"
	@cd .. && ./bin/sdl_example_launcher

# Category targets
nl: $(COMPILER) $(NL_EXAMPLES)
debug: $(COMPILER) $(DEBUG_EXAMPLES)
	@echo ""
	@echo "Debugging & Validation examples built successfully!"
	@echo "Run them:"
	@$(if $(strip $(DEBUG_NAMES)),$(foreach ex,$(DEBUG_NAMES),echo "  ./bin/$(ex)";),:)
	@echo ""
opl: $(COMPILER) $(OPL_EXAMPLES)
	@echo ""
	@echo "OPL DSL toolchain built successfully!"
	@echo "Try: ./bin/opl --help"
sdl: $(COMPILER) $(SDL_EXAMPLES)
ncurses: $(COMPILER) $(NCURSES_EXAMPLES)
opengl: $(COMPILER) $(OPENGL_EXAMPLES)

# Ensure compiler exists
$(COMPILER):
	@cd .. && $(MAKE) bin/nanoc

$(INTERPRETER):
	@cd .. && $(MAKE) bin/nano

# === Build Rules ===

# Debugging & validation examples (in debug/ subdirectory)
$(BIN_DIR)/logging_%: debug/logging_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

$(BIN_DIR)/coverage_demo: debug/coverage_demo.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

$(BIN_DIR)/property_test_%: debug/property_test_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# OPL toolchain CLI (in opl/ subdirectory)
$(BIN_DIR)/opl: $(BIN_DIR)/opl_cli
	@echo "Linking $@..."
	cp $(BIN_DIR)/opl_cli $(BIN_DIR)/opl

$(BIN_DIR)/opl_%: opl/opl_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Nanolang-only examples (now in language/ subdirectory)
$(BIN_DIR)/nl_%: language/nl_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

$(BIN_DIR)/%_repl: language/%_repl.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Advanced examples (in advanced/ subdirectory)
$(BIN_DIR)/%: advanced/%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Data examples (in data/ subdirectory)
$(BIN_DIR)/%: data/%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Network examples (in network/ subdirectory)
$(BIN_DIR)/%: network/%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Physics examples (in physics/ subdirectory)
$(BIN_DIR)/bullet_%: physics/bullet_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Playground examples
$(BIN_DIR)/playground_server: playground/playground_server.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

$(BIN_DIR)/playground: $(BIN_DIR)/playground_server
	@echo "Linking $@..."
	cp $(BIN_DIR)/playground_server $(BIN_DIR)/playground

# Root examples
$(BIN_DIR)/nano_tools_lint: nano_tools_lint.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

$(BIN_DIR)/sdl_example_launcher: sdl_example_launcher.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

$(BIN_DIR)/sdl_asteroids: games/sdl_asteroids.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

$(BIN_DIR)/sdl_checkers: games/sdl_checkers.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

$(BIN_DIR)/sdl_pong: games/sdl_pong.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Graphics (pattern rule for remaining SDL graphics examples)
$(BIN_DIR)/sdl_%: graphics/sdl_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# Audio (pattern rule with fallback)
$(BIN_DIR)/sdl_nanoamp: audio/sdl_nanoamp.nano $(COMPILER)
	@echo "Building SDL NanoAmp - Pixel-perfect Winamp tribute..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/sdl_nanoamp
	@echo "✓ SDL NanoAmp built: $@"

$(BIN_DIR)/sdl_mod_visualizer: audio/sdl_mod_visualizer.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

$(BIN_DIR)/sdl_tracker_shell: audio/sdl_tracker_shell.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

$(BIN_DIR)/sdl_audio_%: audio/sdl_audio_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# NCurses examples (in terminal/ subdirectory)
$(BIN_DIR)/ncurses_%: terminal/ncurses_%.nano $(COMPILER)
	@echo "Building $@..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/$(notdir $@)

# OpenGL examples (in opengl/ subdirectory)
$(BIN_DIR)/opengl_cube: opengl/opengl_cube.nano $(COMPILER)
	@echo "Building OpenGL Cube..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/opengl_cube

$(BIN_DIR)/opengl_teapot: opengl/opengl_teapot.nano $(COMPILER)
	@echo "Building OpenGL Teapot..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/opengl_teapot
	@echo "✓ OpenGL Teapot built: $@"

$(BIN_DIR)/opengl_solar_system: opengl/opengl_solar_system.nano $(COMPILER)
	@echo "Building OpenGL Solar System..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/opengl_solar_system

$(BIN_DIR)/opengl_particle_fountain: opengl/opengl_particle_fountain.nano $(COMPILER)
	@echo "Building OpenGL Particle Fountain..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/opengl_particle_fountain

$(BIN_DIR)/opengl_modern_hello_triangle: opengl/opengl_modern_hello_triangle.nano $(COMPILER)
	@echo "Building Modern OpenGL Hello Triangle..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/opengl_modern_hello_triangle

$(BIN_DIR)/opengl_modern_postprocess: opengl/opengl_modern_postprocess.nano $(COMPILER)
	@echo "Building Modern OpenGL Postprocess..."
	cd .. && NANO_MODULE_PATH=modules ./bin/nanoc examples/$< -o bin/opengl_modern_postprocess

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
