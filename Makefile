CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -Isrc
SANITIZE_FLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
COVERAGE_FLAGS = -fprofile-arcs -ftest-coverage
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
COV_DIR = coverage
RUNTIME_DIR = $(SRC_DIR)/runtime
COMPILER = $(BIN_DIR)/nanoc
INTERPRETER = $(BIN_DIR)/nano
HYBRID_COMPILER = $(BIN_DIR)/nanoc_stage1_5
COMMON_SOURCES = $(SRC_DIR)/lexer.c $(SRC_DIR)/parser.c $(SRC_DIR)/typechecker.c $(SRC_DIR)/eval.c $(SRC_DIR)/transpiler.c $(SRC_DIR)/env.c
COMMON_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(COMMON_SOURCES))
RUNTIME_SOURCES = $(RUNTIME_DIR)/list_int.c $(RUNTIME_DIR)/list_string.c $(RUNTIME_DIR)/list_token.c
RUNTIME_OBJECTS = $(patsubst $(RUNTIME_DIR)/%.c,$(OBJ_DIR)/runtime/%.o,$(RUNTIME_SOURCES))
COMPILER_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/main.o
INTERPRETER_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/tracing.o $(OBJ_DIR)/interpreter_main.o
# Stage 1.5: Hybrid compiler objects (nanolang lexer + C rest)
# Note: Still need lexer.o for utility functions (free_tokens, token_type_name, etc.)
HYBRID_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/lexer_bridge.o $(OBJ_DIR)/lexer_nano.o $(OBJ_DIR)/main_stage1_5.o
PREFIX ?= /usr/local

all: $(COMPILER) $(INTERPRETER)

$(COMPILER): $(COMPILER_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(COMPILER) $(COMPILER_OBJECTS)

$(INTERPRETER): $(INTERPRETER_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -DNANO_INTERPRETER -o $(INTERPRETER) $(INTERPRETER_OBJECTS)

# Stage 1.5: Hybrid compiler with nanolang lexer
$(HYBRID_COMPILER): $(HYBRID_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(HYBRID_COMPILER) $(HYBRID_OBJECTS)

# Compile nanolang lexer to object file (lexer_main.nano -> lexer_nano.o)
$(OBJ_DIR)/lexer_nano.o: src_nano/lexer_main.nano $(COMPILER) | $(OBJ_DIR)
	@echo "Compiling nanolang lexer..."
	$(COMPILER) src_nano/lexer_main.nano -o $(OBJ_DIR)/lexer_nano.tmp --keep-c
	@# Remove main() wrapper from generated C (sed -i works differently on macOS)
	sed -e '/\/\* C main() entry point/,/^}/d' $(OBJ_DIR)/lexer_nano.tmp.c > $(OBJ_DIR)/lexer_nano_noMain.c
	$(CC) $(CFLAGS) -c $(OBJ_DIR)/lexer_nano_noMain.c -o $@
	@rm -f $(OBJ_DIR)/lexer_nano.tmp $(OBJ_DIR)/lexer_nano.tmp.c $(OBJ_DIR)/lexer_nano_noMain.c

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/nanolang.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/runtime/%.o: $(RUNTIME_DIR)/%.c | $(OBJ_DIR)/runtime
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/runtime:
	mkdir -p $(OBJ_DIR)/runtime

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build Stage 1.5 hybrid compiler
stage1.5: $(HYBRID_COMPILER)
	@echo "✓ Stage 1.5 hybrid compiler built: $(HYBRID_COMPILER)"

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/runtime/*.o $(COMPILER) $(INTERPRETER) $(HYBRID_COMPILER) *.out *.out.c tests/*.out tests/*.out.c
	rm -rf .test_output $(COV_DIR)
	rm -f *.gcda *.gcno *.gcov coverage.info
	find . -name "*.gcda" -o -name "*.gcno" | xargs rm -f

test: $(COMPILER) $(INTERPRETER)
	@./test.sh

# Build with AddressSanitizer and UndefinedBehaviorSanitizer
sanitize: CFLAGS += $(SANITIZE_FLAGS)
sanitize: clean $(COMPILER) $(INTERPRETER)
	@echo "Built with sanitizers enabled"

# Build with code coverage instrumentation
coverage: CFLAGS += $(COVERAGE_FLAGS)
coverage: LDFLAGS += $(COVERAGE_FLAGS)
coverage: clean
	@mkdir -p $(COV_DIR)
	$(CC) $(CFLAGS) -c $(COMMON_SOURCES) $(SRC_DIR)/main.c $(SRC_DIR)/interpreter_main.c -Isrc
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(COMPILER) $(COMMON_OBJECTS) $(OBJ_DIR)/main.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(INTERPRETER) $(COMMON_OBJECTS) $(OBJ_DIR)/interpreter_main.o
	@echo "Built with coverage instrumentation"

# Run valgrind memory checks on test suite
valgrind: $(COMPILER) $(INTERPRETER)
	@echo "Running valgrind on test suite..."
	@for example in examples/*.nano; do \
		echo "Checking $$example..."; \
		valgrind --leak-check=full --error-exitcode=1 --quiet \
			$(COMPILER) $$example -o .test_output/valgrind_test 2>&1 | grep -v "Conditional jump" || true; \
	done
	@echo "Valgrind checks complete"

# Generate coverage report
coverage-report: coverage test
	@echo "Generating coverage report..."
	@lcov --capture --directory . --output-file coverage.info
	@lcov --remove coverage.info '/usr/*' --output-file coverage.info
	@genhtml coverage.info --output-directory $(COV_DIR)
	@echo "Coverage report generated in $(COV_DIR)/index.html"

# Install binaries
install: $(COMPILER) $(INTERPRETER)
	install -d $(PREFIX)/bin
	install -m 755 $(COMPILER) $(PREFIX)/bin/nanoc
	install -m 755 $(INTERPRETER) $(PREFIX)/bin/nano
	@echo "Installed to $(PREFIX)/bin"

# Uninstall binaries
uninstall:
	rm -f $(PREFIX)/bin/nanoc
	rm -f $(PREFIX)/bin/nano
	@echo "Uninstalled from $(PREFIX)/bin"

# Run static analysis
lint:
	@echo "Running clang-tidy..."
	@clang-tidy $(SRC_DIR)/*.c -- -Isrc || true

# Quick check (build + test)
check: all test

# Show help
help:
	@echo "nanolang Makefile targets:"
	@echo "  make              - Build compiler and interpreter"
	@echo "  make stage1.5     - Build Stage 1.5 hybrid compiler (nanolang lexer + C)"
	@echo "  make test         - Run test suite"
	@echo "  make sanitize     - Build with memory sanitizers"
	@echo "  make coverage     - Build with coverage instrumentation"
	@echo "  make coverage-report - Generate HTML coverage report"
	@echo "  make valgrind     - Run valgrind memory checks"
	@echo "  make lint         - Run static analysis"
	@echo "  make check        - Build and test"
	@echo "  make install      - Install to $(PREFIX)/bin"
	@echo "  make uninstall    - Uninstall from $(PREFIX)/bin"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help message"

.PHONY: all clean test sanitize coverage coverage-report valgrind install uninstall lint check help stage1.5
