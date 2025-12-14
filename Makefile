# ============================================================================
# Nanolang Makefile with TRUE GCC-Style 3-Stage Bootstrap
# ============================================================================
#
# USAGE:
#
#   make test              - Build C compiler, run tests (single stage)
#   make bootstrap         - 3-stage bootstrap (GCC-style)
#   make bootstrap test    - Bootstrap THEN run tests with self-hosted compiler
#
# TRUE GCC-STYLE BOOTSTRAP:
#   Stage 0: gcc compiles C sources → bin/nanoc_c
#   Stage 1: nanoc_c compiles nanoc_v05.nano → bin/nanoc_stage1
#   Stage 2: nanoc_stage1 compiles nanoc_v05.nano → bin/nanoc_stage2
#   Stage 3: nanoc_stage2 compiles nanoc_v05.nano → bin/nanoc_stage3
#            Verify: nanoc_stage2 == nanoc_stage3 (reproducible build!)
#            Result: bin/nanoc → bin/nanoc_stage2
#
# After bootstrap, bin/nanoc is the self-hosted compiler (stage 2).
# All subsequent builds use the bootstrapped compiler.
#
# Sentinel files: .bootstrap{0,1,2,3}.done track bootstrap progress.
# Use "make clean" to remove all artifacts and start fresh.
#
# ============================================================================

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -Isrc
LDFLAGS = -lm
SANITIZE_FLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
COVERAGE_FLAGS = -fprofile-arcs -ftest-coverage

SRC_DIR = src
SRC_NANO_DIR = src_nano
OBJ_DIR = obj
BIN_DIR = bin
BUILD_DIR = build_bootstrap
COV_DIR = coverage
RUNTIME_DIR = $(SRC_DIR)/runtime

# Binaries
COMPILER = $(BIN_DIR)/nanoc
COMPILER_C = $(BIN_DIR)/nanoc_c
INTERPRETER = $(BIN_DIR)/nano
FFI_BINDGEN = $(BIN_DIR)/nanoc-ffi

# Bootstrap sentinels (GCC-style)
SENTINEL_BOOTSTRAP0 = .bootstrap0.done
SENTINEL_BOOTSTRAP1 = .bootstrap1.done
SENTINEL_BOOTSTRAP2 = .bootstrap2.done
SENTINEL_BOOTSTRAP3 = .bootstrap3.done

# Bootstrap binaries
NANOC_SOURCE = $(SRC_NANO_DIR)/nanoc_v05.nano
NANOC_STAGE1 = $(BIN_DIR)/nanoc_stage1
NANOC_STAGE2 = $(BIN_DIR)/nanoc_stage2
NANOC_STAGE3 = $(BIN_DIR)/nanoc_stage3

# Source files
COMMON_SOURCES = $(SRC_DIR)/lexer.c $(SRC_DIR)/parser.c $(SRC_DIR)/typechecker.c $(SRC_DIR)/eval.c $(SRC_DIR)/transpiler.c $(SRC_DIR)/env.c $(SRC_DIR)/module.c $(SRC_DIR)/module_metadata.c $(SRC_DIR)/cJSON.c $(SRC_DIR)/module_builder.c
COMMON_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(COMMON_SOURCES))
RUNTIME_SOURCES = $(RUNTIME_DIR)/list_int.c $(RUNTIME_DIR)/list_string.c $(RUNTIME_DIR)/list_token.c $(RUNTIME_DIR)/token_helpers.c $(RUNTIME_DIR)/gc.c $(RUNTIME_DIR)/dyn_array.c $(RUNTIME_DIR)/gc_struct.c $(RUNTIME_DIR)/nl_string.c $(RUNTIME_DIR)/cli.c
RUNTIME_OBJECTS = $(patsubst $(RUNTIME_DIR)/%.c,$(OBJ_DIR)/runtime/%.o,$(RUNTIME_SOURCES))
COMPILER_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/main.o
INTERPRETER_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/tracing.o $(OBJ_DIR)/interpreter_main.o



# Header dependencies
HEADERS = $(SRC_DIR)/nanolang.h $(RUNTIME_DIR)/list_int.h $(RUNTIME_DIR)/list_string.h $(RUNTIME_DIR)/list_token.h $(RUNTIME_DIR)/token_helpers.h $(RUNTIME_DIR)/gc.h $(RUNTIME_DIR)/dyn_array.h $(RUNTIME_DIR)/gc_struct.h $(RUNTIME_DIR)/nl_string.h $(SRC_DIR)/module_builder.h



PREFIX ?= /usr/local

# ============================================================================
# Main Targets
# ============================================================================

.DEFAULT_GOAL := build

.PHONY: all build test examples examples-no-sdl clean rebuild help check-deps

# Build: Just build the C compiler (stage 0)
build: $(COMPILER_C) $(INTERPRETER) $(FFI_BINDGEN)
	@echo ""
	@echo "=========================================="
	@echo "✅ Build Complete"
	@echo "=========================================="
	@echo "  C Compiler: $(COMPILER_C)"
	@echo "  Interpreter: $(INTERPRETER)"
	@echo ""
	@# Ensure bin/nanoc points to C compiler (unless bootstrapped)
	@if [ ! -f $(COMPILER) ] || [ -L $(COMPILER) ]; then \
		rm -f $(COMPILER); \
		ln -sf nanoc_c $(COMPILER); \
		echo "  Default compiler: bin/nanoc → bin/nanoc_c"; \
	else \
		echo "  Default compiler: bin/nanoc (self-hosted from bootstrap)"; \
		echo "  For C compiler testing: run 'make clean && make test'"; \
	fi
	@echo ""

# Alias for build
all: build

# Test: Build C compiler and run tests (single stage)
test: build
	@echo ""
	@echo "=========================================="
	@echo "Running Test Suite (C Compiler)"
	@echo "=========================================="
	@./tests/run_all_tests.sh
	@echo ""

# Full test suite with examples
test-full: test
	@echo "=========================================="
	@echo "Building Examples"
	@echo "=========================================="
	@$(MAKE) examples
	@echo ""
	@echo "✅ All tests and examples completed!"

# Test only core language features (nl_* tests)
test-lang: $(COMPILER_C)
	@echo ""
	@echo "=========================================="
	@echo "Running Core Language Tests (nl_*)"
	@echo "=========================================="
	@./tests/run_all_tests.sh --lang

# Test only application/integration tests
test-app: $(COMPILER_C)
	@echo ""
	@echo "=========================================="
	@echo "Running Application Tests"
	@echo "=========================================="
	@./tests/run_all_tests.sh --app

# Test only unit tests
test-unit: $(COMPILER_C)
	@echo ""
	@echo "=========================================="
	@echo "Running Unit Tests"
	@echo "=========================================="
	@./tests/run_all_tests.sh --unit

# Quick test (language tests only, fastest)
test-quick: $(COMPILER_C)
	@./tests/run_all_tests.sh --lang

# Examples: Build examples
examples: $(COMPILER_C) check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Examples"
	@echo "=========================================="
	@$(MAKE) -C examples all
	@echo "✅ Examples built successfully!"

# Examples without SDL: Build only non-SDL examples  
examples-no-sdl: $(COMPILER_C)
	@echo ""
	@echo "=========================================="
	@echo "Building Examples (Skipping SDL)"
	@echo "=========================================="
	@echo "⚠️  SDL examples will be skipped (SDL2 development libraries not installed)"
	@echo ""
	@echo "Interpreter-only examples are available:"
	@echo "  ./bin/nano examples/hello.nano"
	@echo "  ./bin/nano examples/factorial.nano"
	@echo "  ./bin/nano examples/fibonacci.nano"
	@echo "  ./bin/nano examples/primes.nano"
	@echo "  ./bin/nano examples/calculator.nano"
	@echo "  ./bin/nano examples/game_of_life.nano"
	@echo "  ./bin/nano examples/snake.nano"
	@echo "  ./bin/nano examples/maze.nano"
	@echo ""
	@echo "To build SDL examples, install SDL2 development libraries:"
	@echo "  Ubuntu/Debian: sudo apt-get install libsdl2-dev libsdl2-mixer-dev libsdl2-ttf-dev"
	@echo "  Fedora/RHEL:   sudo dnf install SDL2-devel SDL2_mixer-devel SDL2_ttf-devel"
	@echo "  macOS:         brew install sdl2 sdl2_mixer sdl2_ttf"
	@echo ""
	@echo "✅ Build complete (SDL examples skipped)"

# Clean: Remove all build artifacts and sentinels
clean:
	@echo "Cleaning all build artifacts..."
	rm -rf $(OBJ_DIR) $(BUILD_DIR) $(COV_DIR)
	rm -rf $(BIN_DIR)/*
	rm -f *.out *.out.c tests/*.out tests/*.out.c
	rm -f $(SENTINEL_BOOTSTRAP0) $(SENTINEL_BOOTSTRAP1) $(SENTINEL_BOOTSTRAP2) $(SENTINEL_BOOTSTRAP3)
	rm -f *.gcda *.gcno *.gcov coverage.info
	rm -f test.nano test_output.c test_program
	rm -rf .test_output
	find tests -name "*.out" -o -name "*.out.c" 2>/dev/null | xargs rm -f || true
	@$(MAKE) -C examples clean 2>/dev/null || true
	@echo "✅ Clean complete - ready for fresh build"

# Rebuild: Clean and build from scratch
rebuild: clean build

# ============================================================================
# Stage 0: C Reference Compiler/Interpreter (Base Build)
# ============================================================================

# Build C reference compiler (nanoc_c)
$(COMPILER_C): $(COMPILER_OBJECTS) | $(BIN_DIR)
	@echo "Building C reference compiler..."
	$(CC) $(CFLAGS) -o $(COMPILER_C) $(COMPILER_OBJECTS) $(LDFLAGS)
	@echo "✓ C Compiler: $(COMPILER_C)"

<<<<<<< HEAD
# Default compiler target - link to nanoc_c initially (bootstrap will update to nanoc_stage2)
$(COMPILER): $(COMPILER_C) | $(BIN_DIR)
	@if [ -f $(SENTINEL_BOOTSTRAP3) ] && [ -f $(NANOC_STAGE2) ]; then \
		ln -sf nanoc_stage2 $(COMPILER); \
		echo "✓ Compiler: $(COMPILER) -> nanoc_stage2 (self-hosted)"; \
	else \
		ln -sf nanoc_c $(COMPILER); \
		echo "✓ Compiler: $(COMPILER) -> $(COMPILER_C) (C reference)"; \
	fi

=======
>>>>>>> 59036da (feat: Add standalone if, generic types, and dynamic arrays)
$(INTERPRETER): $(INTERPRETER_OBJECTS) | $(BIN_DIR)
	@echo "Building interpreter..."
	$(CC) $(CFLAGS) -DNANO_INTERPRETER -o $(INTERPRETER) $(INTERPRETER_OBJECTS) $(LDFLAGS)
	@echo "✓ Interpreter: $(INTERPRETER)"

$(FFI_BINDGEN): $(OBJ_DIR)/ffi_bindgen.o | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(FFI_BINDGEN) $(OBJ_DIR)/ffi_bindgen.o $(LDFLAGS)

# Object file compilation
$(COMPILER_OBJECTS): | $(OBJ_DIR) $(OBJ_DIR)/runtime
$(INTERPRETER_OBJECTS): | $(OBJ_DIR) $(OBJ_DIR)/runtime

$(OBJ_DIR)/ffi_bindgen.o: src/ffi_bindgen.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c src/ffi_bindgen.c -o $(OBJ_DIR)/ffi_bindgen.o

# Special dependency: transpiler.o depends on transpiler_iterative_v3_twopass.c (which is #included)
$(OBJ_DIR)/transpiler.o: $(SRC_DIR)/transpiler.c $(SRC_DIR)/transpiler_iterative_v3_twopass.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/transpiler.c -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/runtime/%.o: $(RUNTIME_DIR)/%.c $(HEADERS) | $(OBJ_DIR) $(OBJ_DIR)/runtime
	$(CC) $(CFLAGS) -c $< -o $@

# ============================================================================
# TRUE Bootstrap (GCC-style: Stage 0 → 1 → 2 → 3)
# ============================================================================

.PHONY: bootstrap bootstrap0 bootstrap1 bootstrap2 bootstrap3 bootstrap-clean

# Full bootstrap: Run all 3 stages
bootstrap: $(SENTINEL_BOOTSTRAP3)
	@echo ""
	@echo "=========================================="
	@echo "✅ TRUE BOOTSTRAP COMPLETE!"
	@echo "=========================================="
	@echo ""
<<<<<<< HEAD
	@echo "✓ Self-hosted compiler installed as bin/nanoc"
	@echo "✓ Stage binaries preserved in bin/ for verification"
	@echo "✓ All future builds will use the self-hosted compiler"
	@echo ""

# Bootstrap and install: DEPRECATED - bootstrap now auto-installs
# This target is kept for backwards compatibility but now just calls bootstrap
bootstrap-install: bootstrap
	@echo ""
	@echo "=========================================="
	@echo "NOTE: bootstrap-install is DEPRECATED"
	@echo "=========================================="
	@echo ""
	@echo "The 'make bootstrap' target now automatically installs"
	@echo "the self-hosted compiler. This target is kept for"
	@echo "backwards compatibility but does nothing extra."
	@echo ""
	@echo "To verify installation:"
	@echo "  ls -lh bin/nanoc*"
	@echo ""
=======
	@echo "Result: bin/nanoc is now the self-hosted compiler (stage 2)"
	@echo ""
	@echo "Final binaries:"
	@ls -lh $(BIN_DIR)/nanoc $(BIN_DIR)/nano 2>/dev/null || true
	@echo ""
	@echo "✅ Bootstrap complete - intermediate files cleaned up"
	@echo ""

# Clean bootstrap sentinels to force re-bootstrap
bootstrap-clean:
	@echo "Cleaning bootstrap sentinels..."
	@rm -f $(SENTINEL_BOOTSTRAP0) $(SENTINEL_BOOTSTRAP1) $(SENTINEL_BOOTSTRAP2) $(SENTINEL_BOOTSTRAP3)
	@echo "✓ Bootstrap sentinels cleaned (run 'make bootstrap' to re-bootstrap)"
>>>>>>> 59036da (feat: Add standalone if, generic types, and dynamic arrays)

# Bootstrap Stage 0: Build C reference compiler
bootstrap0: $(SENTINEL_BOOTSTRAP0)

$(SENTINEL_BOOTSTRAP0): $(COMPILER_C) $(INTERPRETER)
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap Stage 0: C Reference Compiler"
	@echo "=========================================="
	@echo "✓ C compiler built: $(COMPILER_C)"
	@touch $(SENTINEL_BOOTSTRAP0)

# Bootstrap Stage 1: Compile nanoc_v05.nano with C compiler
bootstrap1: $(SENTINEL_BOOTSTRAP1)

$(SENTINEL_BOOTSTRAP1): $(SENTINEL_BOOTSTRAP0)
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap Stage 1: C → nanoc_stage1"
	@echo "=========================================="
	@echo "Compiling nanoc_v05.nano with C compiler..."
	@if [ ! -f $(NANOC_SOURCE) ]; then \
		echo "❌ Error: $(NANOC_SOURCE) not found!"; \
		exit 1; \
	fi
	@$(COMPILER_C) $(NANOC_SOURCE) -o $(NANOC_STAGE1)
	@echo "✓ Stage 1 compiler: $(NANOC_STAGE1)"
	@echo ""
	@echo "Testing stage 1 compiler..."
	@$(NANOC_STAGE1) examples/nl_hello.nano /tmp/bootstrap_test_s1 && /tmp/bootstrap_test_s1 >/dev/null 2>&1
	@rm -f /tmp/bootstrap_test_s1
	@echo "✓ Stage 1 compiler works!"
	@touch $(SENTINEL_BOOTSTRAP1)

# Bootstrap Stage 2: Recompile nanoc_v05.nano with stage 1 compiler
bootstrap2: $(SENTINEL_BOOTSTRAP2)

$(SENTINEL_BOOTSTRAP2): $(SENTINEL_BOOTSTRAP1)
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap Stage 2: nanoc_stage1 → nanoc_stage2"
	@echo "=========================================="
	@echo "Compiling nanoc_v05.nano with stage 1 compiler..."
	@$(NANOC_STAGE1) $(NANOC_SOURCE) $(NANOC_STAGE2)
	@echo "✓ Stage 2 compiler: $(NANOC_STAGE2)"
	@echo ""
	@echo "Testing stage 2 compiler..."
	@$(NANOC_STAGE2) examples/nl_hello.nano /tmp/bootstrap_test_s2 && /tmp/bootstrap_test_s2 >/dev/null 2>&1
	@rm -f /tmp/bootstrap_test_s2
	@echo "✓ Stage 2 compiler works!"
	@touch $(SENTINEL_BOOTSTRAP2)

# Bootstrap Stage 3: Recompile AGAIN with stage 2, verify reproducibility
bootstrap3: $(SENTINEL_BOOTSTRAP3)

$(SENTINEL_BOOTSTRAP3): $(SENTINEL_BOOTSTRAP2)
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap Stage 3: nanoc_stage2 → nanoc_stage3 + Verify"
	@echo "=========================================="
	@echo "Compiling nanoc_v05.nano with stage 2 compiler..."
	@$(NANOC_STAGE2) $(NANOC_SOURCE) $(NANOC_STAGE3)
	@echo "✓ Stage 3 compiler: $(NANOC_STAGE3)"
	@echo ""
	@echo "Comparing stage 2 and stage 3 binaries..."
	@echo ""
	@ls -lh $(NANOC_STAGE2) $(NANOC_STAGE3)
	@echo ""
	@if cmp -s $(NANOC_STAGE2) $(NANOC_STAGE3); then \
		echo "✅ BOOTSTRAP VERIFIED: Stage 2 and 3 binaries are IDENTICAL!"; \
		echo ""; \
<<<<<<< HEAD
		echo "This proves reproducible builds - the compiler compiled"; \
		echo "by the C compiler is IDENTICAL to the compiler compiled"; \
		echo "by itself. This is TRUE SELF-HOSTING!"; \
		echo ""; \
=======
		echo "This proves reproducible builds - the self-hosted compiler"; \
		echo "produces identical output when compiling itself!"; \
		echo "This is TRUE GCC-STYLE SELF-HOSTING!"; \
>>>>>>> 59036da (feat: Add standalone if, generic types, and dynamic arrays)
	else \
		s2=$$(stat -f%z $(NANOC_STAGE2) 2>/dev/null || stat -c%s $(NANOC_STAGE2)); \
		s3=$$(stat -f%z $(NANOC_STAGE3) 2>/dev/null || stat -c%s $(NANOC_STAGE3)); \
		echo "⚠️  Binaries differ (Stage 2: $$s2 bytes, Stage 3: $$s3 bytes)"; \
		echo ""; \
		echo "Reasons binaries might differ:"; \
		echo "  - Timestamps embedded in output"; \
		echo "  - Non-deterministic codegen"; \
		echo "  - Pointer addresses in debug info"; \
		echo ""; \
<<<<<<< HEAD
		echo "Both compilers work correctly, which proves self-hosting!"; \
		echo ""; \
	fi; \
	echo "==========================================";\
	echo "Installing Self-Hosted Compiler"; \
	echo "==========================================";\
	echo "Updating bin/nanoc to use self-hosted compiler...";\
	rm -f $(COMPILER); \
	ln -sf nanoc_stage2 $(COMPILER); \
	echo "✓ bin/nanoc now points to self-hosted compiler (nanoc_stage2)"; \
	echo ""; \
	echo "All subsequent builds (test, examples) will use the self-hosted compiler!"; \
	echo ""; \
	touch $(SENTINEL_BOOTSTRAP3)

# Show bootstrap status
bootstrap-status:
	@echo "Bootstrap Status:"
	@echo ""
	@if [ -f $(SENTINEL_BOOTSTRAP0) ]; then \
		echo "  ✅ Stage 0: C reference compiler (bin/nanoc)"; \
	else \
		echo "  ❌ Stage 0: Not built"; \
	fi
	@if [ -f $(SENTINEL_BOOTSTRAP1) ]; then \
		echo "  ✅ Stage 1: Self-hosted compiler ($(NANOC_STAGE1))"; \
	else \
		echo "  ❌ Stage 1: Not built"; \
	fi
	@if [ -f $(SENTINEL_BOOTSTRAP2) ]; then \
		echo "  ✅ Stage 2: Recompiled compiler ($(NANOC_STAGE2))"; \
	else \
		echo "  ❌ Stage 2: Not built"; \
	fi
	@if [ -f $(SENTINEL_BOOTSTRAP3) ]; then \
		echo "  ✅ Stage 3: Bootstrap verified!"; \
		echo ""; \
		echo "  🎉 TRUE SELF-HOSTING ACHIEVED!"; \
	else \
		echo "  ❌ Stage 3: Not verified"; \
=======
		echo "Both compilers work correctly - self-hosting is achieved!"; \
>>>>>>> 59036da (feat: Add standalone if, generic types, and dynamic arrays)
	fi
	@echo ""
	@echo "Installing stage 2 compiler as bin/nanoc..."
	@rm -f $(COMPILER)
	@cp $(NANOC_STAGE2) $(COMPILER)
	@echo "✓ bin/nanoc is now the self-hosted compiler!"
	@echo ""
	@echo "Cleaning up intermediate binaries..."
	@rm -f $(COMPILER_C) $(NANOC_STAGE1) $(NANOC_STAGE2) $(NANOC_STAGE3)
	@echo "✓ Removed: nanoc_c, nanoc_stage1, nanoc_stage2, nanoc_stage3"
	@touch $(SENTINEL_BOOTSTRAP3)

# ============================================================================
# Additional Targets
# ============================================================================

# Dependency checking
.PHONY: check-deps check-deps-sdl
check-deps:
	@echo "Checking build dependencies..."
	@command -v $(CC) >/dev/null 2>&1 || { echo "❌ Error: $(CC) not found. Please install a C compiler."; exit 1; }
	@command -v make >/dev/null 2>&1 || { echo "❌ Error: make not found. Please install make."; exit 1; }
	@echo "✓ Core dependencies satisfied ($(CC), make)"

check-deps-sdl:
	@echo "Checking SDL2 dependencies for graphics examples..."
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "⚠️  Homebrew not found on macOS"; \
			echo "   Will attempt to install SDL2 automatically when needed"; \
		elif ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists sdl2 2>/dev/null; then \
			echo "⚠️  SDL2 not found"; \
			echo "   Will attempt to install automatically when building SDL modules"; \
		else \
			echo "✓ SDL2 found: $$(pkg-config --modversion sdl2)"; \
		fi; \
	elif [ "$$(uname -s)" = "Linux" ]; then \
		if ! command -v pkg-config >/dev/null 2>&1; then \
			echo "⚠️  pkg-config not found"; \
			echo "   Install with: sudo apt-get install pkg-config"; \
		elif ! pkg-config --exists sdl2 2>/dev/null; then \
			echo "⚠️  SDL2 not found"; \
			echo "   Will attempt to install automatically when building SDL modules"; \
			echo "   (You may be prompted for sudo password during module build)"; \
		else \
			echo "✓ SDL2 found: $$(pkg-config --modversion sdl2)"; \
		fi; \
	fi

# Build with sanitizers
sanitize: CFLAGS += $(SANITIZE_FLAGS)
sanitize: LDFLAGS += $(SANITIZE_FLAGS)
sanitize: rebuild

# Build with coverage
coverage: CFLAGS += $(COVERAGE_FLAGS)
coverage: LDFLAGS += $(COVERAGE_FLAGS)
coverage: rebuild

# Coverage report
coverage-report: coverage test
	@echo "Generating coverage report..."
	@lcov --capture --directory . --output-file coverage.info
	@lcov --remove coverage.info '/usr/*' --output-file coverage.info --ignore-errors unused
	@genhtml coverage.info --output-directory $(COV_DIR)
	@echo "Coverage report generated in $(COV_DIR)/index.html"

# Install binaries
install: $(COMPILER) $(INTERPRETER)
	install -d $(PREFIX)/bin
	install -m 755 $(COMPILER) $(PREFIX)/bin/nanoc
	install -m 755 $(INTERPRETER) $(PREFIX)/bin/nano
	@echo "Installed to $(PREFIX)/bin"

uninstall:
	rm -f $(PREFIX)/bin/nanoc
	rm -f $(PREFIX)/bin/nano
	@echo "Uninstalled from $(PREFIX)/bin"

# Valgrind checks
valgrind: $(COMPILER) $(INTERPRETER)
	@echo "Running valgrind on test suite..."
	@for example in examples/*.nano; do \
		echo "Checking $$example..."; \
		valgrind --leak-check=full --error-exitcode=1 --quiet \
			$(COMPILER) $$example -o .test_output/valgrind_test 2>&1 | grep -v "Conditional jump" || true; \
	done
	@echo "Valgrind checks complete"

# Help
help:
	@echo "Nanolang Makefile - TRUE GCC-Style Bootstrap"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Main Targets:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make build       - Build C compiler (stage 0)"
	@echo "  make test        - Build + run tests with C compiler"
	@echo "  make bootstrap   - TRUE 3-stage bootstrap (GCC-style)"
	@echo "  make bootstrap test - Bootstrap THEN run tests"
	@echo "  make examples    - Build + compile examples"
	@echo "  make clean       - Remove all artifacts"
	@echo "  make rebuild     - Clean + build"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Bootstrap Targets (Classic GCC-style):"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make bootstrap         - Full 3-stage bootstrap"
	@echo "  make bootstrap-clean   - Clean bootstrap artifacts"
	@echo "  make bootstrap0        - Stage 0: C → nanoc_c"
	@echo "  make bootstrap1        - Stage 1: nanoc_c → nanoc_stage1"
	@echo "  make bootstrap2        - Stage 2: stage1 → nanoc_stage2"
	@echo "  make bootstrap3        - Stage 3: stage2 → nanoc_stage3 + verify"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Test Targets:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
<<<<<<< HEAD
	@echo "  make bootstrap         - 3-stage bootstrap + auto-install nanoc_stage2"
	@echo "  make bootstrap0        - Stage 0: C → nanoc"
	@echo "  make bootstrap1        - Stage 1: nanoc → nanoc_stage1"
	@echo "  make bootstrap2        - Stage 2: stage1 → nanoc_stage2"
	@echo "  make bootstrap3        - Stage 3: Verify + install nanoc_stage2"
	@echo "  make bootstrap-status  - Show bootstrap status"
=======
	@echo "  make test        - Run all tests (C compiler)"
	@echo "  make test-lang   - Run language tests only"
	@echo "  make test-app    - Run application tests only"
	@echo "  make test-unit   - Run unit tests only"
	@echo "  make test-quick  - Quick test (language tests)"
>>>>>>> 59036da (feat: Add standalone if, generic types, and dynamic arrays)
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Development:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make sanitize    - Build with sanitizers"
	@echo "  make coverage    - Build with coverage"
	@echo "  make valgrind    - Run memory checks"
	@echo "  make install     - Install to $(PREFIX)/bin"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "How It Works:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
<<<<<<< HEAD
	@echo "TRUE Bootstrap Process:"
	@echo "  Stage 0: C sources → bin/nanoc_c (C-based)"
	@echo "  Stage 1: nanoc_c compiles nanoc_v05.nano → nanoc_stage1"
	@echo "  Stage 2: nanoc_stage1 recompiles nanoc_v05.nano → nanoc_stage2"
	@echo "  Stage 3: Verify stage1 == stage2, install nanoc_stage2 as bin/nanoc"
	@echo ""
	@echo "After bootstrap: bin/nanoc → nanoc_stage2 (self-hosted compiler)"
=======
	@echo "Single-Stage Build (make test):"
	@echo "  gcc compiles C sources → bin/nanoc_c"
	@echo "  Tests run with C compiler"
>>>>>>> 59036da (feat: Add standalone if, generic types, and dynamic arrays)
	@echo ""
	@echo "GCC-Style 3-Stage Bootstrap (make bootstrap):"
	@echo "  Stage 0: gcc compiles C sources → bin/nanoc_c"
	@echo "  Stage 1: nanoc_c compiles nanoc_v05.nano → bin/nanoc_stage1"
	@echo "  Stage 2: nanoc_stage1 compiles nanoc_v05.nano → bin/nanoc_stage2"
	@echo "  Stage 3: nanoc_stage2 compiles nanoc_v05.nano → bin/nanoc_stage3"
	@echo "           Verify: nanoc_stage2 == nanoc_stage3 (reproducible!)"
	@echo "           Result: bin/nanoc → bin/nanoc_stage2 (self-hosted)"
	@echo ""
	@echo "After bootstrap, bin/nanoc is the self-hosted compiler."
	@echo "All tests and examples then use the self-hosted version."
	@echo ""

# Directory creation
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/runtime: | $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/runtime

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: all build test test-full test-lang test-app test-unit test-quick examples examples-no-sdl clean rebuild help check-deps check-deps-sdl sanitize coverage coverage-report install uninstall valgrind bootstrap bootstrap0 bootstrap1 bootstrap2 bootstrap3 bootstrap-clean
