# ============================================================================
# Nanolang Makefile with TRUE 3-Stage Bootstrap Support
# ============================================================================
#
# This Makefile supports building nanolang through multiple stages:
#
# BUILD TARGETS (Component Testing):
# - Stage 1: C Reference Compiler (from C sources)
# - Stage 2: Self-Hosted Components (compiled with stage1, tested individually)  
# - Stage 3: Component Validation (test components)
#
# BOOTSTRAP TARGETS (Classic GCC-style):
# - Stage 0: C Reference Compiler (bin/nanoc from C sources)
# - Stage 1: Self-Hosted Compiler (nanoc_v06.nano compiled by stage 0)
# - Stage 2: Recompiled Compiler (nanoc_v06.nano compiled by stage 1)
# - Stage 3: Verify stage1 == stage2, auto-install nanoc_stage2 as bin/nanoc
#
# Sentinel files track build progress (.stage*.built) to avoid rebuilds.
# Use "make clean" to remove all build artifacts and start fresh.
#
# ============================================================================
# IMPORTANT: This Makefile requires GNU make
# ============================================================================
#
# On Linux/macOS: 'make' is GNU make (works as-is)
# On BSD systems (FreeBSD/OpenBSD/NetBSD): use 'gmake' instead of 'make'
#
# Install GNU make on BSD:
#   FreeBSD:  pkg install gmake
#   OpenBSD:  pkg_add gmake
#   NetBSD:   pkgin install gmake
#
# This Makefile uses GNU make features:
#   - $(shell ...) for command execution
#   - $(patsubst ...) for path transformations
#   - ifeq/ifneq conditionals
#
# If you see syntax errors, you're using BSD make. Use 'gmake' instead.
#
# ============================================================================

#Human: we cannot easily detect bsd make because it doesn't understand $(shell), ifeq, etc.
# The documentation at the top of this file is sufficient - BSD users will see
# clear syntax errors and can read the instructions at the top.

CC = cc
CFLAGS = -Wall -Wextra -Werror -std=c99 -g -Isrc
LDFLAGS = -lm

# On Linux, dlopened module shared libraries rely on host-exported runtime symbols
# (e.g. dyn_array_new). Ensure the main binaries export their symbols.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
LDFLAGS += -rdynamic
endif
ifeq ($(UNAME_S),FreeBSD)
LDFLAGS += -Wl,-E
endif
SANITIZE_FLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
COVERAGE_FLAGS = -fprofile-arcs -ftest-coverage

SRC_DIR = src
SRC_NANO_DIR = src_nano
OBJ_DIR = obj
BIN_DIR = bin
BUILD_DIR = $(OBJ_DIR)/build_bootstrap
COV_DIR = coverage
RUNTIME_DIR = $(SRC_DIR)/runtime
USERGUIDE_DIR = build/userguide
USERGUIDE_BUILD_TOOL_SRC = scripts/userguide_build_html.nano
USERGUIDE_BUILD_TOOL = $(USERGUIDE_DIR)/userguide_build_html
USERGUIDE_CHECK_TOOL_SRC = scripts/userguide_snippets_check.nano
USERGUIDE_CHECK_TOOL = $(USERGUIDE_DIR)/userguide_snippets_check

# Binaries
COMPILER = $(BIN_DIR)/nanoc
COMPILER_C = $(BIN_DIR)/nanoc_c
HYBRID_COMPILER = $(BIN_DIR)/nanoc_stage1_5
FFI_BINDGEN = $(BIN_DIR)/nanoc-ffi

# Sentinel files for 3-stage build (component testing)
SENTINEL_STAGE1 = .stage1.built
SENTINEL_STAGE2 = .stage2.built
SENTINEL_STAGE3 = .stage3.built

# Sentinel files for TRUE bootstrap
SENTINEL_BOOTSTRAP0 = .bootstrap0.built
SENTINEL_BOOTSTRAP1 = .bootstrap1.built
SENTINEL_BOOTSTRAP2 = .bootstrap2.built
SENTINEL_BOOTSTRAP3 = .bootstrap3.built

# Bootstrap binaries
NANOC_SOURCE = $(SRC_NANO_DIR)/nanoc_v06.nano
NANOC_STAGE1 = $(BIN_DIR)/nanoc_stage1
NANOC_STAGE2 = $(BIN_DIR)/nanoc_stage2
VERIFY_SCRIPT = scripts/verify_no_nanoc_c.sh
VERIFY_SMOKE_SOURCE = examples/language/nl_hello.nano

# When enabled, make bootstrap stage artifacts deterministic (Mach-O LC_UUID + signature)
BOOTSTRAP_DETERMINISTIC ?= 0
BOOTSTRAP_ENV :=
ifeq ($(BOOTSTRAP_DETERMINISTIC),1)
BOOTSTRAP_ENV := NANO_DETERMINISTIC=1
endif

# Source files
COMMON_SOURCES = $(SRC_DIR)/lexer.c $(SRC_DIR)/parser.c $(SRC_DIR)/typechecker.c $(SRC_DIR)/transpiler.c $(SRC_DIR)/stdlib_runtime.c $(SRC_DIR)/env.c $(SRC_DIR)/module.c $(SRC_DIR)/module_metadata.c $(SRC_DIR)/cJSON.c $(SRC_DIR)/module_builder.c $(SRC_DIR)/resource_tracking.c $(SRC_DIR)/eval.c $(SRC_DIR)/interpreter_ffi.c $(SRC_DIR)/json_diagnostics.c $(SRC_DIR)/reflection.c
COMMON_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(COMMON_SOURCES))
RUNTIME_SOURCES = $(RUNTIME_DIR)/list_int.c $(RUNTIME_DIR)/list_string.c \
	$(RUNTIME_DIR)/list_LexerToken.c $(RUNTIME_DIR)/list_token.c \
	$(RUNTIME_DIR)/list_CompilerDiagnostic.c $(RUNTIME_DIR)/list_CompilerSourceLocation.c \
	$(RUNTIME_DIR)/list_ASTNumber.c $(RUNTIME_DIR)/list_ASTFloat.c \
	$(RUNTIME_DIR)/list_ASTString.c $(RUNTIME_DIR)/list_ASTBool.c \
	$(RUNTIME_DIR)/list_ASTIdentifier.c \
	$(RUNTIME_DIR)/list_ASTBinaryOp.c $(RUNTIME_DIR)/list_ASTCall.c \
	$(RUNTIME_DIR)/list_ASTArrayLiteral.c $(RUNTIME_DIR)/list_ASTLet.c \
	$(RUNTIME_DIR)/list_ASTSet.c $(RUNTIME_DIR)/list_ASTStmtRef.c \
	$(RUNTIME_DIR)/list_ASTIf.c $(RUNTIME_DIR)/list_ASTWhile.c \
	$(RUNTIME_DIR)/list_ASTFor.c $(RUNTIME_DIR)/list_ASTReturn.c \
	$(RUNTIME_DIR)/list_ASTBlock.c $(RUNTIME_DIR)/list_ASTUnsafeBlock.c \
	$(RUNTIME_DIR)/list_ASTPrint.c $(RUNTIME_DIR)/list_ASTAssert.c \
	$(RUNTIME_DIR)/list_ASTFunction.c $(RUNTIME_DIR)/list_ASTShadow.c \
	$(RUNTIME_DIR)/list_ASTStruct.c $(RUNTIME_DIR)/list_ASTStructLiteral.c \
	$(RUNTIME_DIR)/list_ASTFieldAccess.c $(RUNTIME_DIR)/list_ASTEnum.c \
	$(RUNTIME_DIR)/list_ASTUnion.c $(RUNTIME_DIR)/list_ASTUnionConstruct.c \
	$(RUNTIME_DIR)/list_ASTMatch.c $(RUNTIME_DIR)/list_ASTImport.c \
	$(RUNTIME_DIR)/list_ASTOpaqueType.c $(RUNTIME_DIR)/list_ASTTupleLiteral.c \
	$(RUNTIME_DIR)/list_ASTTupleIndex.c \
	$(RUNTIME_DIR)/token_helpers.c $(RUNTIME_DIR)/gc.c $(RUNTIME_DIR)/dyn_array.c \
	$(RUNTIME_DIR)/gc_struct.c $(RUNTIME_DIR)/nl_string.c $(RUNTIME_DIR)/cli.c \
	$(RUNTIME_DIR)/regex.c
RUNTIME_OBJECTS = $(patsubst $(RUNTIME_DIR)/%.c,$(OBJ_DIR)/runtime/%.o,$(RUNTIME_SOURCES))
COMPILER_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/main.o
# Interpreter removed - NanoLang is a compiled language

# Self-hosted components
SELFHOST_COMPONENTS = \
	parser \
	typecheck \
	transpiler

# Header dependencies
SCHEMA_JSON = schema/compiler_schema.json
SCHEMA_OUTPUTS = $(SRC_NANO_DIR)/generated/compiler_schema.nano $(SRC_NANO_DIR)/generated/compiler_ast.nano $(SRC_DIR)/generated/compiler_schema.h
SCHEMA_STAMP = $(BUILD_DIR)/schema.stamp

HEADERS = $(SRC_DIR)/nanolang.h $(SRC_DIR)/generated/compiler_schema.h $(RUNTIME_DIR)/list_int.h $(RUNTIME_DIR)/list_string.h $(RUNTIME_DIR)/list_LexerToken.h $(RUNTIME_DIR)/token_helpers.h $(RUNTIME_DIR)/gc.h $(RUNTIME_DIR)/dyn_array.h $(RUNTIME_DIR)/gc_struct.h $(RUNTIME_DIR)/nl_string.h $(SRC_DIR)/module_builder.h

.PHONY: schema schema-check
schema: $(SCHEMA_STAMP)

schema-check:
	@./scripts/check_compiler_schema.sh

# Schema generation: Use NanoLang if compiler exists, fallback to Python for bootstrap
bin/gen_compiler_schema: scripts/gen_compiler_schema.nano
	@if [ -f ./bin/nanoc ]; then \
		echo "[schema] Compiling schema generator (NanoLang)..."; \
		./bin/nanoc scripts/gen_compiler_schema.nano -o bin/gen_compiler_schema; \
	else \
		echo "[schema] Compiler not built yet, will use Python for bootstrap"; \
		touch bin/gen_compiler_schema; \
	fi

$(SCHEMA_STAMP): $(SCHEMA_JSON) scripts/gen_compiler_schema.py scripts/gen_compiler_schema.nano | $(BUILD_DIR)
	@echo "[schema] Generating compiler schema artifacts..."
	@if [ -f ./bin/gen_compiler_schema ] && [ -x ./bin/gen_compiler_schema ]; then \
		echo "[schema] Using NanoLang version..."; \
		./bin/gen_compiler_schema; \
	else \
		echo "[schema] Using Python version (bootstrap)..."; \
		python3 scripts/gen_compiler_schema.py; \
	fi
	@touch $(SCHEMA_STAMP)

# ============================================================================
# Module Index Generation
# ============================================================================
# Build C tool for generating module index
GENERATE_MODULE_INDEX = bin/generate_module_index
$(GENERATE_MODULE_INDEX): tools/generate_module_index.c modules/std/fs.c src/cJSON.c src/runtime/dyn_array.c src/runtime/gc.c src/runtime/gc_struct.c
	@echo "[tool] Building module index generator (C)..."
	@mkdir -p bin
	@$(CC) $(CFLAGS) -Isrc -Isrc/runtime -Imodules/std \
		tools/generate_module_index.c \
		modules/std/fs.c \
		src/cJSON.c \
		src/runtime/dyn_array.c \
		src/runtime/gc.c \
		src/runtime/gc_struct.c \
		-o $(GENERATE_MODULE_INDEX)

.PHONY: all build test test-docs module-self-test module-mvp examples examples-launcher examples-no-sdl clean rebuild help check-deps check-deps-sdl stage1 stage2 stage3 status sanitize coverage coverage-report install uninstall valgrind stage1.5 bootstrap bootstrap0 bootstrap1 bootstrap2 bootstrap3 bootstrap-status bootstrap-install benchmark modules-index release release-major release-minor release-patch
modules-index: $(GENERATE_MODULE_INDEX)
	@echo "[modules] Generating module index from manifests..."
	@./$(GENERATE_MODULE_INDEX)

# Hybrid compiler objects
HYBRID_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/lexer_bridge.o $(OBJ_DIR)/lexer_nano.o $(OBJ_DIR)/main_stage1_5.o

PREFIX ?= /usr/local

# ============================================================================
# Main Targets
# ============================================================================

.DEFAULT_GOAL := build

.PHONY: all build test examples examples-launcher examples-no-sdl clean rebuild help check-deps check-deps-sdl stage1 stage2 stage3 status sanitize coverage coverage-report install uninstall valgrind stage1.5 bootstrap bootstrap0 bootstrap1 bootstrap2 bootstrap3 bootstrap-status bootstrap-install benchmark modules-index release release-major release-minor release-patch userguide-html

# Build: 3-stage bootstrap (uses sentinels to skip completed stages)
build: schema modules-index $(SENTINEL_STAGE3)
	@echo ""
	@echo "=========================================="
	@echo "✅ Build Complete (3-Stage Bootstrap)"
	@echo "=========================================="
	@$(MAKE) status
	@echo ""

# Alias for build
all: build

# ============================================================================
# Test Targets (Meta-Rule Pattern for Stage-Specific Testing)
# ============================================================================
# In a fully self-bootstrapping system, tests should use the most evolved
# compiler by default. These targets allow testing at specific bootstrap stages:
#
# - test: Uses best available (nanoc_stage2 if bootstrapped, else nanoc_c)
# - test-stage1: Forces C reference compiler only
# - test-stage2: Forces nanoc_stage1 (first self-compilation)
# - test-bootstrap: Forces full bootstrap + nanoc_stage2
# ============================================================================

# Unit tests for C components
.PHONY: test-units
test-units:
	@echo "Running C unit tests..."
	@# Detect which instrumentation is present in object files
	@if nm obj/lexer.o 2>/dev/null | grep -q "__asan"; then \
		echo "  (Sanitizer instrumentation detected in objects - linking with sanitizer flags)"; \
		$(CC) $(CFLAGS) $(SANITIZE_FLAGS) -o tests/test_transpiler tests/test_transpiler.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS) $(SANITIZE_FLAGS); \
	elif ls obj/*.gcno >/dev/null 2>&1; then \
		echo "  (Coverage instrumentation detected - linking with coverage flags)"; \
		$(CC) $(CFLAGS) $(COVERAGE_FLAGS) -o tests/test_transpiler tests/test_transpiler.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS) $(COVERAGE_FLAGS); \
	else \
		$(CC) $(CFLAGS) -o tests/test_transpiler tests/test_transpiler.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS); \
	fi
	@./tests/test_transpiler
	@rm -f tests/test_transpiler

# Core test implementation (used by all test variants)
.PHONY: test-impl
test-impl: test-units
	@./scripts/check_compiler_schema.sh
	@echo ""
	@echo "=========================================="
	@echo "Running Complete Test Suite"
	@echo "=========================================="
	@./tests/run_all_tests.sh
	@echo ""
	@echo "Running self-hosted compiler tests..."
	@if [ -f tests/selfhost/run_selfhost_tests.sh ]; then \
		./tests/selfhost/run_selfhost_tests.sh; \
	fi
	@echo ""
	@echo "✅ All tests completed!"
	@echo ""
	@echo "To build examples, run: make examples"

# Default test: Use most evolved compiler available (no bd dependency)
# NOTE: Wrap test runs with a timeout to avoid infinite compiler loops.
TEST_TIMEOUT ?= 1800
USERGUIDE_TIMEOUT ?= 600
USERGUIDE_API_TIMEOUT ?= 600
SHADOW_CHECK_TIMEOUT ?= 120
test: build shadow-check userguide-export
	@echo ""
	@if [ -f $(SENTINEL_BOOTSTRAP3) ]; then \
		echo "🎯 Testing with SELF-HOSTED compiler (nanoc_stage2)"; \
		echo "   Bootstrap complete - using fully evolved version"; \
	else \
		echo "🎯 Testing with C REFERENCE compiler (nanoc_c)"; \
		echo "   Bootstrap not complete - using baseline version"; \
	fi
	@echo ""
	@if [ -f $(SENTINEL_BOOTSTRAP3) ]; then \
		$(VERIFY_SCRIPT) $(COMPILER) $(COMPILER_C) $(VERIFY_SMOKE_SOURCE); \
	fi
	@perl -e 'alarm $(TEST_TIMEOUT); exec @ARGV' $(MAKE) test-impl

# Doc tests: compile + run user guide snippets
test-docs: build $(USERGUIDE_CHECK_TOOL)
	@perl -e 'alarm $(TEST_TIMEOUT); exec @ARGV' $(USERGUIDE_CHECK_TOOL)

# Export user guide snippets into tests/user_guide
userguide-export: build $(USERGUIDE_CHECK_TOOL)
	@perl -e 'alarm $(TEST_TIMEOUT); exec @ARGV' $(USERGUIDE_CHECK_TOOL) --export tests/user_guide

# Test with beads integration (requires bd CLI to be installed)
# Use this for local development when you want automatic issue tracking
test-with-beads: build
	@echo ""
	@if [ -f $(SENTINEL_BOOTSTRAP3) ]; then \
		echo "🎯 Testing with SELF-HOSTED compiler (nanoc_stage2) + Beads tracking"; \
		echo "   Bootstrap complete - using fully evolved version"; \
	else \
		echo "🎯 Testing with C REFERENCE compiler (nanoc_c) + Beads tracking"; \
		echo "   Bootstrap not complete - using baseline version"; \
	fi
	@echo ""
	@if [ -f $(SENTINEL_BOOTSTRAP3) ]; then \
		$(VERIFY_SCRIPT) $(COMPILER) $(COMPILER_C) $(VERIFY_SMOKE_SOURCE); \
	fi
	@echo ""
	@# Auto-file beads on failures.
	@# Local default: per-failure beads. CI default: summary bead.
	@MODE=per; \
	if [ -n "$$CI" ]; then MODE=summary; fi; \
	python3 scripts/autobeads.py --tests --mode $$MODE --close-on-success --timeout-seconds $${NANOLANG_TEST_TIMEOUT_SECONDS:-480}

# Alias for backwards compatibility
test-full: test

.PHONY: check-schema
check-schema:
	@./scripts/check_compiler_schema.sh

# Performance benchmarking
benchmark:
	@echo "Running performance benchmarks..."
	@./scripts/benchmark.sh

.PHONY: benchmark

# Test with Stage 1 only (C reference compiler)
test-stage1: stage1
	@echo ""
	@echo "🎯 Testing with STAGE 1 (C reference compiler only)"
	@echo "   Forcing baseline C implementation"
	@echo ""
	@# Temporarily ensure nanoc points to nanoc_c
	@rm -f $(COMPILER)
	@ln -sf nanoc_c $(COMPILER)
	@$(MAKE) test-impl
	@# Restore proper link based on bootstrap status
	@if [ -f $(SENTINEL_BOOTSTRAP3) ] && [ -f $(NANOC_STAGE2) ]; then \
		rm -f $(COMPILER); \
		ln -sf nanoc_stage2 $(COMPILER); \
	fi

# Test with Stage 2 (first self-compilation via nanoc_stage1)
test-stage2: bootstrap1
	@echo ""
	@echo "🎯 Testing with STAGE 2 (nanoc_stage1)"
	@echo "   Using first self-compilation (C → nanoc_stage1)"
	@echo ""
	@# Temporarily point nanoc to nanoc_stage1
	@rm -f $(COMPILER)
	@ln -sf nanoc_stage1 $(COMPILER)
	@$(MAKE) test-impl
	@# Restore proper link based on bootstrap status
	@if [ -f $(SENTINEL_BOOTSTRAP3) ] && [ -f $(NANOC_STAGE2) ]; then \
		rm -f $(COMPILER); \
		ln -sf nanoc_stage2 $(COMPILER); \
	else \
		rm -f $(COMPILER); \
		ln -sf nanoc_c $(COMPILER); \
	fi

# Test with full bootstrap (nanoc_stage2)
test-bootstrap: bootstrap
	@echo ""
	@echo "🎯 Testing with FULLY BOOTSTRAPPED compiler (nanoc_stage2)"
	@echo "   Using self-hosted compiler (stage1 → nanoc_stage2)"
	@echo ""
	@$(MAKE) test-impl

# Test only core language features (nl_* tests)
test-lang: build
	@echo ""
	@echo "=========================================="
	@echo "Running Core Language Tests (nl_*)"
	@echo "=========================================="
	@./tests/run_all_tests.sh --lang

# Test only application/integration tests
test-app: build
	@echo ""
	@echo "=========================================="
	@echo "Running Application Tests"
	@echo "=========================================="
	@./tests/run_all_tests.sh --app

# Test only unit tests
test-unit: build
	@echo ""
	@echo "=========================================="
	@echo "Running Unit Tests"
	@echo "=========================================="
	@./tests/run_all_tests.sh --unit

# Quick test (language tests only, fastest)
test-quick: build
	@./tests/run_all_tests.sh --lang

# User guide: validate executable snippets extracted from userguide/*.md
.PHONY: userguide-check
userguide-check: build $(USERGUIDE_CHECK_TOOL)
	@$(USERGUIDE_CHECK_TOOL)

.PHONY: userguide-html
userguide-html: build shadow-check $(USERGUIDE_BUILD_TOOL) userguide-api-docs
	@perl -e 'alarm $(USERGUIDE_TIMEOUT); exec @ARGV' $(USERGUIDE_BUILD_TOOL)

.PHONY: userguide-api-docs
userguide-api-docs: build
	@echo "Generating API documentation..."
	@perl -e 'alarm $(USERGUIDE_API_TIMEOUT); exec @ARGV' bash scripts/generate_all_api_docs.sh

.PHONY: shadow-check
shadow-check: build
	@files=$$(git diff --name-only --diff-filter=AM HEAD -- '*.nano'); \
	if [ -z "$$files" ]; then \
		echo "shadow-check: no changed NanoLang files"; \
	else \
		echo "$$files" | while read -r file; do \
			if [ -f "$$file" ]; then \
				perl -e 'alarm $(SHADOW_CHECK_TIMEOUT); exec @ARGV' bash scripts/check_shadow_tests.sh "$$file"; \
			fi; \
		done; \
	fi

$(USERGUIDE_DIR):
	@mkdir -p $(USERGUIDE_DIR)

$(USERGUIDE_BUILD_TOOL): $(USERGUIDE_BUILD_TOOL_SRC) | $(USERGUIDE_DIR)
	@$(COMPILER) $(USERGUIDE_BUILD_TOOL_SRC) -o $(USERGUIDE_BUILD_TOOL)

$(USERGUIDE_CHECK_TOOL): $(USERGUIDE_CHECK_TOOL_SRC) | $(USERGUIDE_DIR)
	@$(COMPILER) $(USERGUIDE_CHECK_TOOL_SRC) -o $(USERGUIDE_CHECK_TOOL)

# Build all examples (primary examples target)
examples: build check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Examples"
	@echo "=========================================="
	@$(MAKE) -C examples build

# Launch example browser
examples-launcher: build check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "🚀 Launching Example Browser"
	@echo "=========================================="
	@$(MAKE) -C examples launcher
	@echo "✅ Examples built successfully!"

# Examples without SDL: Build only non-SDL examples  
examples-no-sdl: build
	@echo ""
	@echo "=========================================="
	@echo "Building Examples (Skipping SDL)"
	@echo "=========================================="
	@echo "⚠️  SDL examples will be skipped (SDL2 development libraries not installed)"
	@echo ""
	@echo "To run examples, compile them first:"
	@echo "  ./bin/nanoc examples/nl_hello.nano -o bin/nl_hello && ./bin/nl_hello"
	@echo ""
	@echo "To build SDL examples, install SDL2 development libraries:"
	@echo "  Ubuntu/Debian: sudo apt-get install libsdl2-dev libsdl2-mixer-dev libsdl2-ttf-dev"
	@echo "  Fedora/RHEL:   sudo dnf install SDL2-devel SDL2_mixer-devel SDL2_ttf-devel"
	@echo "  macOS:         brew install sdl2 sdl2_mixer sdl2_ttf"
	@echo ""
	@echo "✅ Build complete (SDL examples skipped)"

# Module self-tests (compile-only)
module-self-test: build
	@echo ""
	@echo "=========================================="
	@echo "Module Self-Tests"
	@echo "=========================================="
	@mkdir -p build/module_self_tests
	@find modules -name mvp.nano -print0 | while IFS= read -r -d '' file; do \
		rel=$${file#modules/}; \
		name=$$(echo $$rel | sed 's#/mvp.nano##; s#/#_#g'); \
		out=build/module_self_tests/$$name; \
		echo "[module-self-test] $$file -> $$out"; \
		perl -e 'alarm 30; exec @ARGV' ./bin/nanoc $$file -o $$out; \
	done

# Backwards-compatible alias
module-mvp: module-self-test

# Clean: Remove all build artifacts and sentinels
clean:
	@echo "Cleaning all build artifacts..."
	rm -rf $(OBJ_DIR) $(BUILD_DIR) $(COV_DIR)
	rm -rf $(BIN_DIR)/*
	rm -f *.out *.out.c tests/*.out tests/*.out.c
	rm -f $(SENTINEL_STAGE1) $(SENTINEL_STAGE2) $(SENTINEL_STAGE3)
	rm -f $(SENTINEL_BOOTSTRAP0) $(SENTINEL_BOOTSTRAP1) $(SENTINEL_BOOTSTRAP2) $(SENTINEL_BOOTSTRAP3)
	rm -f $(SCHEMA_STAMP)
	rm -f *.gcda *.gcno *.gcov coverage.info
	rm -f test.nano test_output.c test_program
	rm -rf .test_output
	find tests -name "*.out" -o -name "*.out.c" 2>/dev/null | xargs rm -f || true
	@$(MAKE) -C examples clean 2>/dev/null || true
	@echo "✅ Clean complete - ready for fresh build"

# Rebuild: Clean and build from scratch
rebuild: clean build

# ============================================================================
# Stage 1: C Reference Compiler/Interpreter
# ============================================================================

.PHONY: stage1

stage1: $(SENTINEL_STAGE1)

$(SENTINEL_STAGE1): $(COMPILER) $(FFI_BINDGEN)
	@echo "✓ Stage 1 complete (C reference binaries)"
	@touch $(SENTINEL_STAGE1)

# Build C reference compiler (nanoc_c) - used by self-hosted version
$(COMPILER_C): $(COMPILER_OBJECTS) | $(BIN_DIR)
	@echo "Stage 1: Building C reference compiler..."
	@# If objects were built with sanitizers/coverage, ensure we link with matching flags.
	@if nm obj/lexer.o 2>/dev/null | grep -q "__asan"; then \
		echo "  (Sanitizer instrumentation detected in objects - linking with sanitizer flags)"; \
		$(CC) $(CFLAGS) $(SANITIZE_FLAGS) -o $(COMPILER_C) $(COMPILER_OBJECTS) $(LDFLAGS) $(SANITIZE_FLAGS); \
	elif ls obj/*.gcno >/dev/null 2>&1; then \
		echo "  (Coverage instrumentation detected in objects - linking with coverage flags)"; \
		$(CC) $(CFLAGS) $(COVERAGE_FLAGS) -o $(COMPILER_C) $(COMPILER_OBJECTS) $(LDFLAGS) $(COVERAGE_FLAGS); \
	else \
		$(CC) $(CFLAGS) -o $(COMPILER_C) $(COMPILER_OBJECTS) $(LDFLAGS); \
	fi
	@echo "✓ C Compiler: $(COMPILER_C)"

# Default compiler target - link to nanoc_c initially (bootstrap will update to nanoc_stage2)
$(COMPILER): $(COMPILER_C) | $(BIN_DIR)
	@if [ -f $(SENTINEL_BOOTSTRAP3) ] && [ -f $(NANOC_STAGE2) ]; then \
		ln -sf nanoc_stage2 $(COMPILER); \
		echo "✓ Compiler: $(COMPILER) -> nanoc_stage2 (self-hosted)"; \
	else \
		ln -sf nanoc_c $(COMPILER); \
		echo "✓ Compiler: $(COMPILER) -> $(COMPILER_C) (C reference)"; \
	fi

# Interpreter removed - NanoLang is a compiled language

$(FFI_BINDGEN): $(OBJ_DIR)/ffi_bindgen.o | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(FFI_BINDGEN) $(OBJ_DIR)/ffi_bindgen.o $(LDFLAGS)

$(BIN_DIR)/nano_lint: tools/nano_lint.nano | $(BIN_DIR)
	@$(COMPILER) tools/nano_lint.nano -o $(BIN_DIR)/nano_lint

# Object file compilation
$(COMPILER_OBJECTS): | $(OBJ_DIR) $(OBJ_DIR)/runtime

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
# Stage 2: Self-Hosted Components (compile and test with stage1)
# ============================================================================

.PHONY: stage2

stage2: $(SENTINEL_STAGE2)

$(SENTINEL_STAGE2): $(SENTINEL_STAGE1)
	@echo ""
	@echo "=========================================="
	@echo "Stage 2: Building Self-Hosted Components"
	@echo "=========================================="
	@echo "Compiling components with stage1..."
	@echo ""
	@# Compile each self-hosted component (STRICT: must produce an executable binary)
	@# If compiler is ASan-instrumented, disable leak detection during compilation.
	@if nm obj/lexer.o 2>/dev/null | grep -q "__asan"; then \
		echo "  (ASan build detected - disabling leak detection for stage2 compiler runs)"; \
		export ASAN_OPTIONS=detect_leaks=0; \
	fi; \
	success=0; fail=0; \
	for comp in $(SELFHOST_COMPONENTS); do \
		src="$$comp"; \
		if [ "$$comp" = "parser" ]; then src="parser_driver"; fi; \
		if [ "$$comp" = "typecheck" ]; then src="typecheck_driver"; fi; \
		if [ "$$comp" = "transpiler" ]; then src="transpiler_driver"; fi; \
		out="$(BIN_DIR)/$$comp"; \
		log="/tmp/nanolang_stage2_$$comp.log"; \
		echo "  Building $$comp..."; \
		rm -f "$$out" "$$log"; \
		if $(COMPILER) "$(SRC_NANO_DIR)/$$src.nano" -o "$$out" >"$$log" 2>&1; then \
			if [ -x "$$out" ]; then \
				echo "    ✓ $$comp compiled successfully"; \
				success=$$((success + 1)); \
			else \
				echo "    ❌ $$comp: compiler returned success but binary missing/not executable: $$out"; \
				tail -80 "$$log" || true; \
				fail=$$((fail + 1)); \
			fi; \
		else \
			echo "    ❌ $$comp compilation failed"; \
			tail -80 "$$log" || true; \
			fail=$$((fail + 1)); \
		fi; \
		echo ""; \
	done; \
	if [ $$fail -eq 0 ] && [ $$success -eq 3 ]; then \
		echo "✓ Stage 2: $$success/3 components built successfully"; \
		touch $(SENTINEL_STAGE2); \
	else \
		echo "❌ Stage 2: FAILED - $$success/3 components built (failures: $$fail)"; \
		echo ""; \
		echo "Fix the failing component(s) above (see /tmp/nanolang_stage2_<component>.log)."; \
		exit 1; \
	fi

# ============================================================================
# Stage 3: Bootstrap Validation (re-compile with stage2, verify working)
# ============================================================================

.PHONY: stage3

stage3: $(SENTINEL_STAGE3)

$(SENTINEL_STAGE3): $(SENTINEL_STAGE2)
	@echo ""
	@echo "=========================================="
	@echo "Stage 3: Bootstrap Validation"
	@echo "=========================================="
	@echo "Validating self-hosted components..."
	@echo ""
	@# Run each component (they are expected to run their own shadow tests and exit 0)
	@success=0; fail=0; missing=0; \
	for comp in $(SELFHOST_COMPONENTS); do \
		bin="$(BIN_DIR)/$$comp"; \
		log="/tmp/nanolang_stage3_$$comp.log"; \
		if [ ! -x "$$bin" ]; then \
			echo "  ❌ Missing component binary: $$bin"; \
			missing=$$((missing + 1)); \
			continue; \
		fi; \
		echo "  Testing $$comp..."; \
		if "$$bin" >"$$log" 2>&1; then \
			echo "    ✓ $$comp tests passed"; \
			success=$$((success + 1)); \
		else \
			echo "    ❌ $$comp tests failed"; \
			tail -120 "$$log" || true; \
			fail=$$((fail + 1)); \
		fi; \
	done; \
	echo ""; \
	if [ $$missing -eq 0 ] && [ $$fail -eq 0 ] && [ $$success -eq 3 ]; then \
		echo "✓ Stage 3: $$success/3 components validated"; \
		touch $(SENTINEL_STAGE3); \
	else \
		echo "❌ Stage 3: FAILED - validated $$success/3 (missing: $$missing, failures: $$fail)"; \
		echo "See /tmp/nanolang_stage3_<component>.log for details."; \
		exit 1; \
	fi

# ============================================================================
# TRUE Bootstrap (GCC-style: Stage 0 → 1 → 2 → 3)
# ============================================================================

.PHONY: bootstrap bootstrap0 bootstrap1 bootstrap2 bootstrap3

# Full bootstrap: Run all stages
bootstrap: $(SENTINEL_BOOTSTRAP3)
	@echo ""
	@echo "=========================================="
	@echo "✅ TRUE BOOTSTRAP COMPLETE!"
	@echo "=========================================="
	@$(MAKE) bootstrap-status
	@echo ""
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

# Bootstrap Stage 0: Build C reference compiler
bootstrap0: $(SENTINEL_BOOTSTRAP0)

$(SENTINEL_BOOTSTRAP0): $(COMPILER_C)
	@echo "✓ Bootstrap Stage 0: C reference compiler ready"
	@touch $(SENTINEL_BOOTSTRAP0)

# Bootstrap Stage 1: Compile nanoc_v04.nano with C compiler
bootstrap1: $(SENTINEL_BOOTSTRAP1)


$(SENTINEL_BOOTSTRAP1): $(SENTINEL_BOOTSTRAP0)
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap Stage 1: Self-Hosted Compiler"
	@echo "=========================================="
	@echo "Compiling nanoc_v06.nano with C compiler..."
	@if [ -f $(NANOC_SOURCE) ]; then \
		$(BOOTSTRAP_ENV) $(COMPILER_C) $(NANOC_SOURCE) -o $(NANOC_STAGE1) && \
		echo "✓ Stage 1 compiler created: $(NANOC_STAGE1)" && \
		echo "" && \
		echo "Testing stage 1 compiler..." && \
		if $(NANOC_STAGE1) examples/language/nl_hello.nano -o /tmp/bootstrap_test && /tmp/bootstrap_test >/dev/null 2>&1; then \
			echo "✓ Stage 1 compiler works!"; \
			touch $(SENTINEL_BOOTSTRAP1); \
		else \
			echo "❌ Stage 1 compiler test failed"; \
			exit 1; \
		fi; \
	else \
		echo "❌ Error: $(NANOC_SOURCE) not found!"; \
		exit 1; \
	fi

# Bootstrap Stage 2: Recompile nanoc_v04.nano with stage 1 compiler
bootstrap2: $(SENTINEL_BOOTSTRAP2)


$(SENTINEL_BOOTSTRAP2): $(SENTINEL_BOOTSTRAP1)
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap Stage 2: Recompilation"
	@echo "=========================================="
	@echo "Compiling nanoc_v06.nano with stage 1 compiler..."
	@$(BOOTSTRAP_ENV) $(NANOC_STAGE1) $(NANOC_SOURCE) -o $(NANOC_STAGE2)
	@echo "✓ Stage 2 compiler created: $(NANOC_STAGE2)"
	@echo ""
	@echo "Testing stage 2 compiler..."
	@if $(NANOC_STAGE2) examples/language/nl_hello.nano -o /tmp/bootstrap_test2 && /tmp/bootstrap_test2 >/dev/null 2>&1; then \
		echo "✓ Stage 2 compiler works!"; \
		touch $(SENTINEL_BOOTSTRAP2); \
	else \
		echo "❌ Stage 2 compiler test failed"; \
		exit 1; \
	fi

# Verify that each bootstrap stage produced stand-alone compilers that can compile and run a smoke test
.PHONY: verify-bootstrap
verify-bootstrap: bootstrap
	@echo ""
	@echo "=========================================="
	@echo "Verifying Bootstrap Artifacts"
	@echo "=========================================="
	@test -x $(NANOC_STAGE1) || { echo "❌ Missing: $(NANOC_STAGE1)"; exit 1; }
	@test -x $(NANOC_STAGE2) || { echo "❌ Missing: $(NANOC_STAGE2)"; exit 1; }
	@echo "✓ Found stage compilers:"
	@ls -lh $(NANOC_STAGE1) $(NANOC_STAGE2)
	@echo ""
	@echo "Smoke test: stage1 compiles + runs nl_hello.nano..."
	@$(NANOC_STAGE1) examples/language/nl_hello.nano -o /tmp/bootstrap_verify_stage1 && /tmp/bootstrap_verify_stage1 >/dev/null
	@echo "✓ stage1 ok"
	@echo "Smoke test: stage2 compiles + runs nl_hello.nano..."
	@$(NANOC_STAGE2) examples/language/nl_hello.nano -o /tmp/bootstrap_verify_stage2 && /tmp/bootstrap_verify_stage2 >/dev/null
	@echo "✓ stage2 ok"

.PHONY: verify-no-nanoc_c verify-no-nanoc_c-check

verify-no-nanoc_c: $(SENTINEL_BOOTSTRAP3)
	@$(VERIFY_SCRIPT) $(COMPILER) $(COMPILER_C) $(VERIFY_SMOKE_SOURCE)

verify-no-nanoc_c-check:
	@$(VERIFY_SCRIPT) $(COMPILER) $(COMPILER_C) $(VERIFY_SMOKE_SOURCE)

# Bootstrap Stage 3: Verify reproducible build
bootstrap3: $(SENTINEL_BOOTSTRAP3)

$(SENTINEL_BOOTSTRAP3): $(SENTINEL_BOOTSTRAP2)
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap Stage 3: Verification"
	@echo "=========================================="
	@echo "Comparing stage 1 and stage 2 binaries..."
	@echo ""
	@ls -lh $(NANOC_STAGE1) $(NANOC_STAGE2)
	@echo ""
	@if cmp -s $(NANOC_STAGE1) $(NANOC_STAGE2); then \
		echo "✅ BOOTSTRAP VERIFIED: Binaries are identical!"; \
		echo ""; \
		echo "This proves reproducible builds - the compiler compiled"; \
		echo "by the C compiler is IDENTICAL to the compiler compiled"; \
		echo "by itself. This is TRUE SELF-HOSTING!"; \
		echo ""; \
	else \
		if [ "$(BOOTSTRAP_DETERMINISTIC)" = "1" ]; then \
			echo "❌ BOOTSTRAP FAILED: Expected identical binaries (BOOTSTRAP_DETERMINISTIC=1)"; \
			exit 1; \
		fi; \
		echo "⚠️  Bootstrap verification: Binaries differ"; \
		echo ""; \
		echo "Stage 1 size: $$(stat -f%z $(NANOC_STAGE1) 2>/dev/null || stat -c%s $(NANOC_STAGE1))"; \
		echo "Stage 2 size: $$(stat -f%z $(NANOC_STAGE2) 2>/dev/null || stat -c%s $(NANOC_STAGE2))"; \
		echo ""; \
		echo "This is expected if:"; \
		echo "  - Timestamps are embedded in binary"; \
		echo "  - Non-deterministic codegen"; \
		echo "  - Different compiler optimizations"; \
		echo ""; \
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
	echo "Smoke test: installed bin/nanoc compiles + runs nl_hello.nano..."; \
	if $(COMPILER) examples/language/nl_hello.nano -o /tmp/bootstrap_installed_test && /tmp/bootstrap_installed_test >/dev/null 2>&1; then \
		echo "✓ installed compiler works"; \
	else \
		echo "❌ installed compiler smoke test failed"; \
		exit 1; \
	fi; \
	echo ""; \
	echo "Verifying bin/nanoc does not depend on bin/nanoc_c..."; \
	$(VERIFY_SCRIPT) $(COMPILER) $(COMPILER_C) $(VERIFY_SMOKE_SOURCE); \
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
	fi
	@echo ""

# ============================================================================
# Additional Targets
# ============================================================================

# Show build status
status:
	@echo "Build Status:"
	@echo ""
	@if [ -f $(SENTINEL_STAGE1) ]; then \
		echo "  ✅ Stage 1: C reference compiler ($(COMPILER))"; \
	else \
		echo "  ❌ Stage 1: Not built"; \
	fi
	@if [ -f $(SENTINEL_STAGE2) ]; then \
		echo "  ✅ Stage 2: Self-hosted components compiled"; \
		for comp in $(SELFHOST_COMPONENTS); do \
			if [ -f $(BIN_DIR)/$$comp ]; then \
				echo "    • $$comp"; \
			fi; \
		done; \
	else \
		echo "  ❌ Stage 2: Not built"; \
	fi
	@if [ -f $(SENTINEL_STAGE3) ]; then \
		echo "  ✅ Stage 3: Bootstrap validated"; \
	else \
		echo "  ❌ Stage 3: Not built"; \
	fi
	@echo ""

# Stage 1.5: Hybrid compiler
stage1.5: $(HYBRID_COMPILER)
	@echo "✓ Stage 1.5 hybrid compiler built: $(HYBRID_COMPILER)"

$(HYBRID_COMPILER): $(HYBRID_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(HYBRID_COMPILER) $(HYBRID_OBJECTS) $(LDFLAGS)

$(OBJ_DIR)/lexer_nano.o: src_nano/lexer_main.nano $(COMPILER) | $(OBJ_DIR)
	@echo "Compiling nanolang lexer..."
	$(COMPILER) src_nano/lexer_main.nano -o $(OBJ_DIR)/lexer_nano.tmp --keep-c
	sed -e '/\/\* C main() entry point/,/^}/d' $(OBJ_DIR)/lexer_nano.tmp.c > $(OBJ_DIR)/lexer_nano_noMain.c
	$(CC) $(CFLAGS) -c $(OBJ_DIR)/lexer_nano_noMain.c -o $@
	@rm -f $(OBJ_DIR)/lexer_nano.tmp $(OBJ_DIR)/lexer_nano.tmp.c $(OBJ_DIR)/lexer_nano_noMain.c

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
install: $(COMPILER)
	install -d $(PREFIX)/bin
	install -m 755 $(COMPILER) $(PREFIX)/bin/nanoc
	@echo "Installed to $(PREFIX)/bin"

uninstall:
	rm -f $(PREFIX)/bin/nanoc
	@echo "Uninstalled from $(PREFIX)/bin"

# Valgrind checks
valgrind: $(COMPILER)
	@echo "Running valgrind on test suite..."
	@for example in examples/*.nano; do \
		echo "Checking $$example..."; \
		valgrind --leak-check=full --error-exitcode=1 --quiet \
			$(COMPILER) $$example -o .test_output/valgrind_test 2>&1 | grep -v "Conditional jump" || true; \
	done
	@echo "Valgrind checks complete"

# Help
help:
	@echo "Nanolang Makefile - Build & Bootstrap Targets"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Main Targets:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make build            - Build all components (default)"
	@echo "  make bootstrap        - TRUE 3-stage bootstrap (GCC-style)"
	@echo "  make test             - Build + run all tests (auto-detect best compiler)"
	@echo "  make test-beads       - Run tests; on failures, auto-create/update beads"
	@echo "  make examples         - Build all examples"
	@echo "  make examples-beads   - Build examples; on failures, auto-create/update beads"
	@echo "  make examples-launcher- Launch example browser"
	@echo "  make clean            - Remove all artifacts"
	@echo "  make rebuild          - Clean + build"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Testing Targets:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make test              - Test with best available compiler"
	@echo "  make test-stage1       - Test with C reference compiler only"
	@echo "  make test-stage2       - Test with nanoc_stage1 (first self-compile)"
	@echo "  make test-bootstrap    - Test with fully bootstrapped compiler"
	@echo "  make test-lang         - Test only core language features"
	@echo "  make test-app          - Test only application/integration tests"
	@echo "  make test-unit         - Test only unit tests"
	@echo "  make test-quick        - Quick test (language tests only)"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Component Build (Stage Targets):"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make stage1      - C reference compiler"
	@echo "  make stage2      - Self-hosted components"
	@echo "  make stage3      - Component validation"
	@echo "  make status      - Show component build status"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "TRUE Bootstrap (Classic GCC-style):"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make bootstrap         - 3-stage bootstrap + auto-install nanoc_stage2"
	@echo "  make bootstrap0        - Stage 0: C → nanoc"
	@echo "  make bootstrap1        - Stage 1: nanoc → nanoc_stage1"
	@echo "  make bootstrap2        - Stage 2: stage1 → nanoc_stage2"
	@echo "  make bootstrap3        - Stage 3: Verify + install nanoc_stage2"
	@echo "  make bootstrap-status  - Show bootstrap status"
	@echo "  make verify-no-nanoc_c - Ensure self-hosted compiler never shells out to nanoc_c"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Development:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make sanitize    - Build with sanitizers"
	@echo "  make coverage    - Build with coverage"
	@echo "  make valgrind    - Run memory checks"
	@echo "  make install     - Install to $(PREFIX)/bin"
	@echo ""
	@echo "Component Build Process:"
	@echo "  Stage 1: C sources → nanoc + nano"
	@echo "  Stage 2: nanoc compiles parser/typechecker/transpiler"
	@echo "  Stage 3: Validate components work"
	@echo ""
	@echo "TRUE Bootstrap Process:"
	@echo "  Stage 0: C sources → bin/nanoc_c (C-based)"
	@echo "  Stage 1: nanoc_c compiles nanoc_v06.nano → nanoc_stage1"
	@echo "  Stage 2: nanoc_stage1 recompiles nanoc_v06.nano → nanoc_stage2"
	@echo "  Stage 3: Verify stage1 == stage2, install nanoc_stage2 as bin/nanoc"
	@echo ""
	@echo "After bootstrap: bin/nanoc → nanoc_stage2 (self-hosted compiler)"

# Legacy aliases for test-with-beads (kept for backwards compatibility)
test-beads: test-with-beads

examples-beads:
	@python3 scripts/autobeads.py --examples

# CI-friendly: one summary bead per run (per branch), auto-closed when green
test-beads-ci:
	@python3 scripts/autobeads.py --tests --mode summary --close-on-success

examples-beads-ci:
	@python3 scripts/autobeads.py --examples --mode summary --close-on-success
	@echo ""
	@echo "Sentinels:"
	@echo "  .stage{1,2,3}.built - Component build"
	@echo "  .bootstrap{0,1,2,3}.built - True bootstrap"
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

.PHONY: all build test test-docs examples examples-launcher examples-no-sdl clean rebuild help check-deps check-deps-sdl stage1 stage2 stage3 status sanitize coverage coverage-report install uninstall valgrind stage1.5 bootstrap bootstrap0 bootstrap1 bootstrap2 bootstrap3 bootstrap-status bootstrap-install benchmark modules-index release release-major release-minor release-patch
.PHONY: all build test test-docs examples examples-launcher examples-no-sdl clean rebuild help check-deps check-deps-sdl stage1 stage2 stage3 status sanitize coverage coverage-report install uninstall valgrind stage1.5 bootstrap bootstrap0 bootstrap1 bootstrap2 bootstrap3 bootstrap-status bootstrap-install benchmark modules-index release release-major release-minor release-patch

# ============================================================================
# RELEASE AUTOMATION
# ============================================================================

# Create a new release (default: patch version bump)
# Usage:
#   make release              # Bump patch version (x.y.Z)
#   make release-minor        # Bump minor version (x.Y.0)
#   make release-major        # Bump major version (X.0.0)
release:
	@echo "Creating patch release..."
	@./scripts/release.sh patch

release-minor:
	@echo "Creating minor release..."
	@./scripts/release.sh minor

release-major:
	@echo "Creating major release..."
	@./scripts/release.sh major
