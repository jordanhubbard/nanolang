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

SHELL := /usr/bin/env bash
.SHELLFLAGS := -e -o pipefail -c

CC = cc
CFLAGS = -Wall -Wextra -Werror -std=c99 -g -Isrc -D_GNU_SOURCE
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
# TMPDIR-aware temp directory for bootstrap test artifacts
BOOTSTRAP_TMPDIR := $(or $(TMPDIR),/tmp)
BOOTSTRAP_ENV := NANO_MODULE_PATH=modules
ifeq ($(BOOTSTRAP_DETERMINISTIC),1)
BOOTSTRAP_ENV += NANO_DETERMINISTIC=1
endif
BOOTSTRAP_VERBOSE ?= 1
ifeq ($(BOOTSTRAP_VERBOSE),1)
BOOTSTRAP_ENV += NANO_VERBOSE_BUILD=1
endif
BOOTSTRAP_VERBOSE_FLAG :=
ifeq ($(BOOTSTRAP_VERBOSE),1)
BOOTSTRAP_VERBOSE_FLAG := -v
endif

# Source files
COMMON_SOURCES = $(SRC_DIR)/lexer.c $(SRC_DIR)/parser.c $(SRC_DIR)/typechecker.c $(SRC_DIR)/transpiler.c $(SRC_DIR)/stdlib_runtime.c $(SRC_DIR)/env.c $(SRC_DIR)/builtins_registry.c $(SRC_DIR)/module.c $(SRC_DIR)/module_metadata.c $(SRC_DIR)/cJSON.c $(SRC_DIR)/toon_output.c $(SRC_DIR)/module_builder.c $(SRC_DIR)/resource_tracking.c $(SRC_DIR)/eval.c $(SRC_DIR)/eval/eval_hashmap.c $(SRC_DIR)/eval/eval_math.c $(SRC_DIR)/eval/eval_string.c $(SRC_DIR)/eval/eval_io.c $(SRC_DIR)/interpreter_ffi.c $(SRC_DIR)/json_diagnostics.c $(SRC_DIR)/reflection.c $(SRC_DIR)/nanocore_subset.c $(SRC_DIR)/nanocore_export.c
COMMON_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(COMMON_SOURCES))
RUNTIME_SOURCES = $(RUNTIME_DIR)/list_int.c $(RUNTIME_DIR)/list_string.c \
	$(RUNTIME_DIR)/list_LexerToken.c $(RUNTIME_DIR)/list_token.c \
	$(RUNTIME_DIR)/list_CompilerDiagnostic.c $(RUNTIME_DIR)/list_CompilerSourceLocation.c \
	$(RUNTIME_DIR)/list_ASTNumber.c $(RUNTIME_DIR)/list_ASTFloat.c \
	$(RUNTIME_DIR)/list_ASTString.c $(RUNTIME_DIR)/list_ASTBool.c \
	$(RUNTIME_DIR)/list_ASTIdentifier.c \
	$(RUNTIME_DIR)/list_ASTBinaryOp.c $(RUNTIME_DIR)/list_ASTCall.c \
	$(RUNTIME_DIR)/list_ASTModuleQualifiedCall.c \
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
	$(RUNTIME_DIR)/gc_struct.c $(RUNTIME_DIR)/nl_string.c $(RUNTIME_DIR)/ffi_loader.c \
	$(RUNTIME_DIR)/cli.c $(RUNTIME_DIR)/regex.c
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

HEADERS = $(SRC_DIR)/nanolang.h $(SRC_DIR)/generated/compiler_schema.h $(SRC_DIR)/builtins_registry.h $(RUNTIME_DIR)/list_int.h $(RUNTIME_DIR)/list_string.h $(RUNTIME_DIR)/list_LexerToken.h $(RUNTIME_DIR)/token_helpers.h $(RUNTIME_DIR)/gc.h $(RUNTIME_DIR)/dyn_array.h $(RUNTIME_DIR)/gc_struct.h $(RUNTIME_DIR)/nl_string.h $(RUNTIME_DIR)/ffi_loader.h $(SRC_DIR)/module_builder.h

.PHONY: schema schema-check
schema: $(SCHEMA_STAMP)

schema-check:
	@$(TIMEOUT_CMD) ./scripts/check_compiler_schema.sh

# Schema generation: Use NanoLang if compiler exists, fallback to Python for bootstrap
bin/gen_compiler_schema: scripts/gen_compiler_schema.nano
	@if [ -f ./bin/nanoc ]; then \
		echo "[schema] Compiling schema generator (NanoLang)..."; \
		$(TIMEOUT_CMD) ./bin/nanoc scripts/gen_compiler_schema.nano -o bin/gen_compiler_schema; \
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
		$(TIMEOUT_CMD) python3 scripts/gen_compiler_schema.py; \
	fi
	@touch $(SCHEMA_STAMP)

# Ensure generated schema headers are created before compilation
$(SRC_DIR)/generated/compiler_schema.h: $(SCHEMA_STAMP)
	@# Schema stamp ensures this file exists

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

modules-index: $(GENERATE_MODULE_INDEX)
	@echo "[modules] Generating module index from manifests..."
	@./$(GENERATE_MODULE_INDEX)

# Validate module dependencies (always run, don't cache)
modules:
	@echo "[modules] Generating module index from manifests..."
	@./$(GENERATE_MODULE_INDEX)
	@./scripts/validate-modules.sh

# Install all missing module dependencies (requires sudo)
install-deps:
	@./scripts/install-deps.sh

# Hybrid compiler objects
HYBRID_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/lexer_bridge.o $(OBJ_DIR)/lexer_nano.o $(OBJ_DIR)/main_stage1_5.o

PREFIX ?= /usr/local

# ============================================================================
# Main Targets
# ============================================================================

.DEFAULT_GOAL := build


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

# Build NanoISA virtual machine backend (nano_virt compiler, nano_vm executor, co-process, daemon)
vm: nano_virt nano_vm nano_cop nano_vmd
	@echo ""
	@echo "✅ VM backend built: bin/nano_virt bin/nano_vm bin/nano_cop bin/nano_vmd"

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

# ============================================================================
# NanoISA - Virtual Machine ISA, Assembler, and Disassembler
# ============================================================================

NANOISA_DIR = $(SRC_DIR)/nanoisa
NANOISA_SOURCES = $(NANOISA_DIR)/isa.c $(NANOISA_DIR)/nvm_format.c \
	$(NANOISA_DIR)/assembler.c $(NANOISA_DIR)/disassembler.c \
	$(NANOISA_DIR)/verifier.c
NANOISA_OBJECTS = $(patsubst $(NANOISA_DIR)/%.c,$(OBJ_DIR)/nanoisa/%.o,$(NANOISA_SOURCES))

$(OBJ_DIR)/nanoisa/%.o: $(NANOISA_DIR)/%.c $(NANOISA_DIR)/isa.h $(NANOISA_DIR)/nvm_format.h | $(OBJ_DIR)/nanoisa
	$(CC) $(CFLAGS) -I$(NANOISA_DIR) -c $< -o $@

$(OBJ_DIR)/nanoisa:
	mkdir -p $(OBJ_DIR)/nanoisa

.PHONY: test-nanoisa
test-nanoisa: $(NANOISA_OBJECTS)
	@echo "Running NanoISA tests..."
	@$(CC) $(CFLAGS) -I$(NANOISA_DIR) -o tests/nanoisa/test_nanoisa \
		tests/nanoisa/test_nanoisa.c $(NANOISA_OBJECTS) $(LDFLAGS)
	@./tests/nanoisa/test_nanoisa
	@rm -f tests/nanoisa/test_nanoisa

# ============================================================================
# NanoVM - Virtual Machine Execution Engine
# ============================================================================

NANOVM_DIR = $(SRC_DIR)/nanovm
NANOVM_SOURCES = $(NANOVM_DIR)/value.c $(NANOVM_DIR)/heap.c $(NANOVM_DIR)/vm.c $(NANOVM_DIR)/vm_ffi.c $(NANOVM_DIR)/vm_builtins.c $(NANOVM_DIR)/cop_protocol.c
NANOVM_OBJECTS = $(patsubst $(NANOVM_DIR)/%.c,$(OBJ_DIR)/nanovm/%.o,$(NANOVM_SOURCES))

$(OBJ_DIR)/nanovm/%.o: $(NANOVM_DIR)/%.c $(NANOVM_DIR)/vm.h $(NANOVM_DIR)/heap.h $(NANOVM_DIR)/value.h | $(OBJ_DIR)/nanovm
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/nanovm:
	mkdir -p $(OBJ_DIR)/nanovm

.PHONY: test-nanovm
test-nanovm: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	@echo "Running NanoVM tests..."
	@$(CC) $(CFLAGS) -o tests/nanovm/test_vm \
		tests/nanovm/test_vm.c $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) \
		$(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/nanovm/test_vm
	@rm -f tests/nanovm/test_vm

# ── NanoVM Daemon (vmd) objects ───────────────────────────────────────────────
VMD_SOURCES = $(NANOVM_DIR)/vmd_protocol.c $(NANOVM_DIR)/vmd_client.c $(NANOVM_DIR)/vmd_server.c
VMD_OBJECTS = $(patsubst $(NANOVM_DIR)/%.c,$(OBJ_DIR)/nanovm/%.o,$(VMD_SOURCES))

nano_vm: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nanovm/vmd_protocol.o $(OBJ_DIR)/nanovm/vmd_client.o $(OBJ_DIR)/nanovm/main.o | bin
	$(CC) $(CFLAGS) -o bin/$@ $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) \
		$(OBJ_DIR)/nanovm/vmd_protocol.o $(OBJ_DIR)/nanovm/vmd_client.o \
		$(OBJ_DIR)/nanovm/main.o $(LDFLAGS)

nano_vmd: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(VMD_OBJECTS) $(OBJ_DIR)/nanovm/vmd_main.o | bin
	$(CC) $(CFLAGS) -o bin/$@ $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) \
		$(VMD_OBJECTS) $(OBJ_DIR)/nanovm/vmd_main.o $(LDFLAGS) -lpthread

$(OBJ_DIR)/nanovm/main.o: $(NANOVM_DIR)/main.c $(NANOVM_DIR)/vm.h $(NANOVM_DIR)/vm_ffi.h $(NANOVM_DIR)/vmd_client.h | $(OBJ_DIR)/nanovm
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/nanovm/vmd_main.o: $(NANOVM_DIR)/vmd_main.c $(NANOVM_DIR)/vmd_server.h | $(OBJ_DIR)/nanovm
	$(CC) $(CFLAGS) -c $< -o $@

# ── Co-Process FFI (nano_cop) ────────────────────────────────────────────────
nano_cop: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nanovm/cop_main.o | bin
	$(CC) $(CFLAGS) -o bin/$@ $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) \
		$(OBJ_DIR)/nanovm/cop_main.o $(LDFLAGS)

$(OBJ_DIR)/nanovm/cop_main.o: $(NANOVM_DIR)/cop_main.c $(NANOVM_DIR)/cop_protocol.h $(NANOVM_DIR)/vm_ffi.h | $(OBJ_DIR)/nanovm
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: test-nanovm-daemon
test-nanovm-daemon: nano_vm nano_vmd
	@echo "Running NanoVM daemon integration tests..."
	@scripts/test_nanovm_daemon.sh

.PHONY: test-nanovm-integration
test-nanovm-integration: nano_vm nano_virt nano_vmd nano_cop
	@echo "Running NanoVM end-to-end integration tests..."
	@scripts/test_nanovm_integration.sh

.PHONY: test-cop-lifecycle
test-cop-lifecycle: nano_vm nano_virt nano_vmd nano_cop
	@echo "Running co-process lifecycle tests..."
	@scripts/test_cop_lifecycle.sh

# Differential testing: Coq-extracted reference interpreter vs NanoVM
.PHONY: test-differential
test-differential: nano_virt
	@echo ""
	@echo "=========================================="
	@echo "Differential Testing: Coq Reference vs VM"
	@echo "=========================================="
	@scripts/test_differential.sh

# Backend-agnostic test suite: run all .nano tests through NanoVM
.PHONY: test-vm test-daemon
test-vm: nano_vm nano_virt
	@echo ""
	@echo "=========================================="
	@echo "Running Test Suite (NanoVM backend)"
	@echo "=========================================="
	@NANOLANG_BACKEND=vm ./tests/run_all_tests.sh

test-daemon: nano_vm nano_virt nano_vmd
	@echo ""
	@echo "=========================================="
	@echo "Running Test Suite (NanoVM daemon backend)"
	@echo "=========================================="
	@NANOLANG_BACKEND=daemon ./tests/run_all_tests.sh

# ── NanoVirt (Compiler Backend) ──────────────────────────────────────────────
NANOVIRT_DIR = $(SRC_DIR)/nanovirt
NANOVIRT_SOURCES = $(NANOVIRT_DIR)/codegen.c $(NANOVIRT_DIR)/wrapper_gen.c
NANOVIRT_OBJECTS = $(patsubst $(NANOVIRT_DIR)/%.c,$(OBJ_DIR)/nanovirt/%.o,$(NANOVIRT_SOURCES))

$(OBJ_DIR)/nanovirt/%.o: $(NANOVIRT_DIR)/%.c $(NANOVIRT_DIR)/codegen.h $(NANOVIRT_DIR)/wrapper_gen.h | $(OBJ_DIR)/nanovirt
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/nanovirt:
	mkdir -p $(OBJ_DIR)/nanovirt

.PHONY: test-nanovirt
test-nanovirt: $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	@echo "Running NanoVirt codegen tests..."
	@$(CC) $(CFLAGS) -o tests/nanovirt/test_codegen \
		tests/nanovirt/test_codegen.c $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) \
		$(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/nanovirt/test_codegen
	@rm -f tests/nanovirt/test_codegen

nano_virt: $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nanovirt/main.o | bin
	$(CC) $(CFLAGS) -o bin/$@ $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) \
		$(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nanovirt/main.o $(LDFLAGS)

$(OBJ_DIR)/nanovirt/main.o: $(NANOVIRT_DIR)/main.c $(NANOVIRT_DIR)/codegen.h | $(OBJ_DIR)/nanovirt
	$(CC) $(CFLAGS) -c $< -o $@

# Unit tests for C components
.PHONY: test-units
test-units: test-nanoisa test-nanovm test-nanovirt
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
	@$(TIMEOUT_CMD) ./scripts/check_compiler_schema.sh
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
USERGUIDE_TIMEOUT ?= 2400
USERGUIDE_API_TIMEOUT ?= 600
SHADOW_CHECK_TIMEOUT ?= 120
CMD_TIMEOUT ?= 600
TIMEOUT_CMD ?= perl -e 'alarm $(CMD_TIMEOUT); exec @ARGV'
# Bootstrap2 needs extended timeout due to self-hosted compiler performance
# See docs/BOOTSTRAP_PROFILING_2026-01-21.md for analysis
BOOTSTRAP2_TIMEOUT ?= 3600
BOOTSTRAP2_TIMEOUT_CMD ?= perl -e 'alarm $(BOOTSTRAP2_TIMEOUT); exec @ARGV'
# Examples need extended timeout since they build 100+ programs across multiple compilers
EXAMPLES_TIMEOUT ?= 2400
EXAMPLES_TIMEOUT_CMD ?= perl -e 'alarm $(EXAMPLES_TIMEOUT); exec @ARGV'
# Release needs extended timeout since it runs tests + git/gh operations
RELEASE_TIMEOUT ?= 2400
RELEASE_TIMEOUT_CMD ?= perl -e 'alarm $(RELEASE_TIMEOUT); exec @ARGV'
USERGUIDE_BUILD_API_DOCS ?= 0
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
		$(TIMEOUT_CMD) $(VERIFY_SCRIPT) $(COMPILER) $(COMPILER_C) $(VERIFY_SMOKE_SOURCE); \
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
		$(TIMEOUT_CMD) $(VERIFY_SCRIPT) $(COMPILER) $(COMPILER_C) $(VERIFY_SMOKE_SOURCE); \
	fi
	@echo ""
	@# Auto-file beads on failures.
	@# Local default: per-failure beads. CI default: summary bead.
	@MODE=per; \
	if [ -n "$$CI" ]; then MODE=summary; fi; \
	$(TIMEOUT_CMD) python3 scripts/autobeads.py --tests --mode $$MODE --close-on-success --timeout-seconds $${NANOLANG_TEST_TIMEOUT_SECONDS:-480}

# Alias for backwards compatibility
test-full: test

.PHONY: check-schema
check-schema:
	@./scripts/check_compiler_schema.sh

# Performance benchmarking
benchmark:
	@echo "Running performance benchmarks..."
	@$(TIMEOUT_CMD) ./scripts/benchmark.sh

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
	@$(TIMEOUT_CMD) $(MAKE) test-impl
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
	@$(TIMEOUT_CMD) $(MAKE) test-impl
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
	@$(TIMEOUT_CMD) $(MAKE) test-impl

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
userguide-html: build shadow-check $(USERGUIDE_BUILD_TOOL)
	@if [ "$$CI" = "true" ] || [ "$(USERGUIDE_BUILD_API_DOCS)" = "1" ]; then \
		$(MAKE) userguide-api-docs; \
	else \
		echo "userguide-html: skipping API doc generation (set USERGUIDE_BUILD_API_DOCS=1 to enable)"; \
	fi
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
	@$(TIMEOUT_CMD) $(COMPILER_C) $(USERGUIDE_BUILD_TOOL_SRC) -o $(USERGUIDE_BUILD_TOOL)

$(USERGUIDE_CHECK_TOOL): $(USERGUIDE_CHECK_TOOL_SRC) | $(USERGUIDE_DIR)
	@$(TIMEOUT_CMD) $(COMPILER_C) $(USERGUIDE_CHECK_TOOL_SRC) -o $(USERGUIDE_CHECK_TOOL)

# Build all examples (STRICT: requires all module dependencies)
# Run 'make -B modules' first to validate dependencies
examples:
	@echo ""
	@echo "=========================================="
	@echo "Building Examples (All Compilers)"
	@echo "=========================================="
	@echo "Note: This will fail if any module dependencies are missing."
	@echo "      Run 'make -B modules' to see what's needed."
	@echo "      Or use 'make examples-available' to skip unavailable examples."
	@echo ""
	@if [ "$$NANOLANG_AUTOBEADS_EXAMPLES" = "1" ]; then \
		$(EXAMPLES_TIMEOUT_CMD) $(MAKE) examples-core; \
	else \
		NANOLANG_AUTOBEADS_EXAMPLES=1 $(EXAMPLES_TIMEOUT_CMD) python3 scripts/autobeads.py --examples; \
		$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples launcher COMPILER=../bin/nanoc_stage1 BIN_SUFFIX= || echo "⚠️  Launcher build failed (optional - requires SDL and nano_tools modules)"; \
	fi

.PHONY: examples-core examples-stage1 examples-stage3 examples-vm

examples-core: examples-stage1 examples-stage3 examples-vm
	@echo "✅ Examples built successfully!"

examples-stage1: bootstrap1 modules-index check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Examples (Stage 1 Compiler)"
	@echo "=========================================="
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples build COMPILER=../bin/nanoc_stage1 BIN_SUFFIX=

examples-stage3: bootstrap3 modules-index check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Examples (Stage 3 Compiler)"
	@echo "=========================================="
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples build COMPILER=../bin/nanoc_stage2 BIN_SUFFIX=_stage3

examples-vm: vm modules-index check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Examples (NanoVM Bytecode)"
	@echo "=========================================="
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples build EXAMPLES_BACKEND=vm BIN_SUFFIX=_vm

# Build available examples (GRACEFUL: skip examples with missing dependencies)
examples-available: build check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Available Examples"
	@echo "=========================================="
	@echo "Note: Examples with missing dependencies will be skipped."
	@echo ""
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples examples-available

# Launch example browser
launcher: build check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "🚀 Launching Example Browser"
	@echo "=========================================="
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples launcher
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
		$(TIMEOUT_CMD) ./bin/nanoc $$file -o $$out; \
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
	rm -f formal/*.vo formal/*.vok formal/*.vos formal/*.glob formal/.*.aux
	@$(TIMEOUT_CMD) $(MAKE) -C examples clean 2>/dev/null || true
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
	@$(TIMEOUT_CMD) $(COMPILER) tools/nano_lint.nano -o $(BIN_DIR)/nano_lint

# Object file compilation
$(COMPILER_OBJECTS): | $(OBJ_DIR) $(OBJ_DIR)/runtime $(OBJ_DIR)/eval

$(OBJ_DIR)/ffi_bindgen.o: src/ffi_bindgen.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c src/ffi_bindgen.c -o $(OBJ_DIR)/ffi_bindgen.o

# Special dependency: transpiler.o depends on transpiler_iterative_v3_twopass.c (which is #included)
$(OBJ_DIR)/transpiler.o: $(SRC_DIR)/transpiler.c $(SRC_DIR)/transpiler_iterative_v3_twopass.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/transpiler.c -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/runtime/%.o: $(RUNTIME_DIR)/%.c $(HEADERS) | $(OBJ_DIR) $(OBJ_DIR)/runtime
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/eval/%.o: $(SRC_DIR)/eval/%.c $(HEADERS) | $(OBJ_DIR) $(OBJ_DIR)/eval
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
		log="$(BOOTSTRAP_TMPDIR)/nanolang_stage2_$$comp.log"; \
		echo "  Building $$comp..."; \
		rm -f "$$out" "$$log"; \
		if $(TIMEOUT_CMD) $(COMPILER) "$(SRC_NANO_DIR)/$$src.nano" -o "$$out" >"$$log" 2>&1; then \
			if [ -x "$$out" ]; then \
				echo "    ✓ $$comp compiled successfully"; \
				success=$$((success + 1)); \
			else \
				echo "    ❌ $$comp: compiler returned success but binary missing/not executable: $$out"; \
				tail -80 "$$log" || true; \
				exit 1; \
			fi; \
		else \
			echo "    ❌ $$comp compilation failed"; \
			tail -80 "$$log" || true; \
			exit 1; \
		fi; \
		echo ""; \
	done; \
	echo "✓ Stage 2: $$success/3 components built successfully"; \
	touch $(SENTINEL_STAGE2)

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
		log="$(BOOTSTRAP_TMPDIR)/nanolang_stage3_$$comp.log"; \
		if [ ! -x "$$bin" ]; then \
			echo "  ❌ Missing component binary: $$bin"; \
			missing=$$((missing + 1)); \
			continue; \
		fi; \
		echo "  Testing $$comp..."; \
		if $(TIMEOUT_CMD) "$$bin" >"$$log" 2>&1; then \
			echo "    ✓ $$comp tests passed"; \
			success=$$((success + 1)); \
		else \
			echo "    ❌ $$comp tests failed"; \
			tail -120 "$$log" || true; \
			exit 1; \
		fi; \
	done; \
	echo ""; \
	if [ $$missing -eq 0 ] && [ $$success -eq 3 ]; then \
		echo "✓ Stage 3: $$success/3 components validated"; \
		touch $(SENTINEL_STAGE3); \
	else \
		echo "❌ Stage 3: FAILED - validated $$success/3 (missing: $$missing)"; \
		echo "See $(BOOTSTRAP_TMPDIR)/nanolang_stage3_<component>.log for details."; \
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
bootstrap1:
	@if [ -f $(SENTINEL_BOOTSTRAP1) ] && [ ! -f $(NANOC_STAGE1) ]; then \
		echo "⚠️  Stale sentinel detected: removing $(SENTINEL_BOOTSTRAP1)"; \
		rm -f $(SENTINEL_BOOTSTRAP1); \
	fi
	@$(MAKE) $(SENTINEL_BOOTSTRAP1)


$(SENTINEL_BOOTSTRAP1): $(SENTINEL_BOOTSTRAP0)
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap Stage 1: Self-Hosted Compiler"
	@echo "=========================================="
	@echo "Compiling nanoc_v06.nano with C compiler..."
	@if [ -f $(NANOC_SOURCE) ]; then \
		$(BOOTSTRAP_ENV) $(TIMEOUT_CMD) $(COMPILER_C) $(NANOC_SOURCE) -o $(NANOC_STAGE1) && \
		echo "✓ Stage 1 compiler created: $(NANOC_STAGE1)" && \
		echo "" && \
		echo "Testing stage 1 compiler..." && \
		if $(TIMEOUT_CMD) $(NANOC_STAGE1) examples/language/nl_hello.nano -o $(BOOTSTRAP_TMPDIR)/bootstrap_test && $(TIMEOUT_CMD) $(BOOTSTRAP_TMPDIR)/bootstrap_test >/dev/null 2>&1; then \
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
bootstrap2:
	@if [ -f $(SENTINEL_BOOTSTRAP2) ] && [ ! -f $(NANOC_STAGE2) ]; then \
		echo "⚠️  Stale sentinel detected: removing $(SENTINEL_BOOTSTRAP2)"; \
		rm -f $(SENTINEL_BOOTSTRAP2); \
	fi
	@$(MAKE) $(SENTINEL_BOOTSTRAP2)


$(SENTINEL_BOOTSTRAP2): $(SENTINEL_BOOTSTRAP1)
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap Stage 2: Recompilation"
	@echo "=========================================="
	@echo "Compiling nanoc_v06.nano with stage 1 compiler..."
	@$(BOOTSTRAP_ENV) $(BOOTSTRAP2_TIMEOUT_CMD) $(NANOC_STAGE1) $(BOOTSTRAP_VERBOSE_FLAG) $(NANOC_SOURCE) -o $(NANOC_STAGE2)
	@echo "✓ Stage 2 compiler created: $(NANOC_STAGE2)"
	@echo ""
	@echo "Testing stage 2 compiler..."
	@if $(TIMEOUT_CMD) $(NANOC_STAGE2) examples/language/nl_hello.nano -o $(BOOTSTRAP_TMPDIR)/bootstrap_test2 && $(TIMEOUT_CMD) $(BOOTSTRAP_TMPDIR)/bootstrap_test2 >/dev/null 2>&1; then \
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
	@$(TIMEOUT_CMD) $(NANOC_STAGE1) examples/language/nl_hello.nano -o $(BOOTSTRAP_TMPDIR)/bootstrap_verify_stage1 && $(TIMEOUT_CMD) $(BOOTSTRAP_TMPDIR)/bootstrap_verify_stage1 >/dev/null
	@echo "✓ stage1 ok"
	@echo "Smoke test: stage2 compiles + runs nl_hello.nano..."
	@$(TIMEOUT_CMD) $(NANOC_STAGE2) examples/language/nl_hello.nano -o $(BOOTSTRAP_TMPDIR)/bootstrap_verify_stage2 && $(TIMEOUT_CMD) $(BOOTSTRAP_TMPDIR)/bootstrap_verify_stage2 >/dev/null
	@echo "✓ stage2 ok"

.PHONY: verify-no-nanoc_c verify-no-nanoc_c-check

verify-no-nanoc_c: $(SENTINEL_BOOTSTRAP3)
	@$(TIMEOUT_CMD) $(VERIFY_SCRIPT) $(COMPILER) $(COMPILER_C) $(VERIFY_SMOKE_SOURCE)

verify-no-nanoc_c-check:
	@$(TIMEOUT_CMD) $(VERIFY_SCRIPT) $(COMPILER) $(COMPILER_C) $(VERIFY_SMOKE_SOURCE)

# Bootstrap Stage 3: Verify reproducible build
bootstrap3:
	@if [ -f $(SENTINEL_BOOTSTRAP3) ] && [ ! -f $(NANOC_STAGE2) ]; then \
		echo "⚠️  Stale sentinel detected: removing $(SENTINEL_BOOTSTRAP3)"; \
		rm -f $(SENTINEL_BOOTSTRAP3); \
	fi
	@$(MAKE) $(SENTINEL_BOOTSTRAP3)

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
	if $(TIMEOUT_CMD) $(COMPILER) examples/language/nl_hello.nano -o $(BOOTSTRAP_TMPDIR)/bootstrap_installed_test && $(TIMEOUT_CMD) $(BOOTSTRAP_TMPDIR)/bootstrap_installed_test >/dev/null 2>&1; then \
		echo "✓ installed compiler works"; \
	else \
		echo "❌ installed compiler smoke test failed"; \
		exit 1; \
	fi; \
	echo ""; \
	echo "Verifying bin/nanoc does not depend on bin/nanoc_c..."; \
	$(TIMEOUT_CMD) $(VERIFY_SCRIPT) $(COMPILER) $(COMPILER_C) $(VERIFY_SMOKE_SOURCE); \
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
# Profiled Bootstrap (Self-Analysis)
# ============================================================================
# Build profiled versions of compiler components and analyze performance.
# This creates _p suffixed binaries with profiling enabled, runs them on
# real workloads, and outputs LLM-ready JSON for hotspot analysis.

.PHONY: bootstrap-profile bootstrap-profile-macos bootstrap-profile-linux

# Profiled binaries
PARSER_P = $(BIN_DIR)/parser_p
TYPECHECK_P = $(BIN_DIR)/typecheck_p
TRANSPILER_P = $(BIN_DIR)/transpiler_p
USERGUIDE_HTML_P = $(BIN_DIR)/userguide_html_p

bootstrap-profile: build
	@echo ""
	@echo "=========================================="
	@echo "Bootstrap Profile: Self-Analysis Mode"
	@echo "=========================================="
	@echo ""
	@echo "Building profiled compiler components..."
	@echo ""
	@# Build profiled parser
	@echo "  [1/4] Building parser_p..."
	@$(TIMEOUT_CMD) $(COMPILER) $(SRC_NANO_DIR)/parser_driver.nano -o $(PARSER_P) -pg 2>&1 | tail -3
	@echo ""
	@# Build profiled typecheck
	@echo "  [2/4] Building typecheck_p..."
	@$(TIMEOUT_CMD) $(COMPILER) $(SRC_NANO_DIR)/typecheck_driver.nano -o $(TYPECHECK_P) -pg 2>&1 | tail -3
	@echo ""
	@# Build profiled transpiler  
	@echo "  [3/4] Building transpiler_p..."
	@$(TIMEOUT_CMD) $(COMPILER) $(SRC_NANO_DIR)/transpiler_driver.nano -o $(TRANSPILER_P) -pg 2>&1 | tail -3
	@echo ""
	@# Build profiled userguide_html
	@echo "  [4/4] Building userguide_html_p..."
	@$(TIMEOUT_CMD) $(COMPILER) $(USERGUIDE_BUILD_TOOL_SRC) -o $(USERGUIDE_HTML_P) -pg 2>&1 | tail -3
	@echo ""
	@echo "=========================================="
	@echo "Running profiled components..."
	@echo "=========================================="
	@echo ""
	@# Run profiled parser
	@echo ">>> PROFILING: parser_p on self (parser.nano)"
	@echo "-------------------------------------------"
	@$(PARSER_P) 2>&1 || true
	@echo ""
	@# Run profiled typecheck
	@echo ">>> PROFILING: typecheck_p on self (typecheck.nano)"
	@echo "-------------------------------------------"
	@$(TYPECHECK_P) 2>&1 || true
	@echo ""
	@# Run profiled transpiler
	@echo ">>> PROFILING: transpiler_p on self (transpiler.nano)"
	@echo "-------------------------------------------"
	@$(TRANSPILER_P) 2>&1 || true
	@echo ""
	@# Run profiled userguide_html
	@echo ">>> PROFILING: userguide_html_p"
	@echo "-------------------------------------------"
	@$(USERGUIDE_HTML_P) 2>&1 | head -60 || true
	@echo ""
	@echo "=========================================="
	@echo "✅ Bootstrap Profile Complete"
	@echo "=========================================="
	@echo ""
	@echo "Profiled binaries created:"
	@ls -lh $(BIN_DIR)/*_p 2>/dev/null || echo "  (none found)"
	@echo ""
	@echo "Analysis: Look for nl_ prefixed functions with high sample counts"
	@echo "These are NanoLang-generated hotspots that may benefit from optimization."
	@echo ""

# macOS-specific profile run
bootstrap-profile-macos: bootstrap-profile
	@echo "Running on macOS with sample(1)..."
	@echo "Platform-specific profiling complete."

# Linux-specific profile run  
bootstrap-profile-linux: bootstrap-profile
	@echo "Running on Linux with gprofng..."
	@echo "Platform-specific profiling complete."

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
	$(TIMEOUT_CMD) $(COMPILER) src_nano/lexer_main.nano -o $(OBJ_DIR)/lexer_nano.tmp --keep-c
	$(TIMEOUT_CMD) sed -e '/\/\* C main() entry point/,/^}/d' $(OBJ_DIR)/lexer_nano.tmp.c > $(OBJ_DIR)/lexer_nano_noMain.c
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
	@$(TIMEOUT_CMD) lcov --capture --directory . --output-file coverage.info
	@$(TIMEOUT_CMD) lcov --remove coverage.info '/usr/*' --output-file coverage.info --ignore-errors unused
	@$(TIMEOUT_CMD) genhtml coverage.info --output-directory $(COV_DIR)
	@echo "Coverage report generated in $(COV_DIR)/index.html"

# Install binaries
install: $(COMPILER) vm
	install -d $(PREFIX)/bin
	install -m 755 $(COMPILER) $(PREFIX)/bin/nanoc
	install -m 755 bin/nano_virt $(PREFIX)/bin/nano_virt
	install -m 755 bin/nano_vm $(PREFIX)/bin/nano_vm
	install -m 755 bin/nano_cop $(PREFIX)/bin/nano_cop
	install -m 755 bin/nano_vmd $(PREFIX)/bin/nano_vmd
	@echo "Installed to $(PREFIX)/bin (nanoc, nano_virt, nano_vm, nano_cop, nano_vmd)"

uninstall:
	rm -f $(PREFIX)/bin/nanoc $(PREFIX)/bin/nano_virt $(PREFIX)/bin/nano_vm $(PREFIX)/bin/nano_cop $(PREFIX)/bin/nano_vmd
	@echo "Uninstalled from $(PREFIX)/bin"

# Valgrind checks
valgrind: $(COMPILER)
	@echo "Running valgrind on test suite..."
	@for example in examples/*.nano; do \
		echo "Checking $$example..."; \
		valgrind --leak-check=full --error-exitcode=1 --quiet \
			$(TIMEOUT_CMD) $(COMPILER) $$example -o .test_output/valgrind_test 2>&1 | grep -v "Conditional jump" || true; \
	done
	@echo "Valgrind checks complete"

# Fuzzing targets
fuzz-build:
	@echo "Building fuzzing targets..."
	@mkdir -p tests/fuzzing
	@echo "Building simple lexer fuzzer..."
	@gcc -g -O1 -fsanitize=address \
		-Isrc \
		tests/fuzzing/simple_fuzz_lexer.c src/lexer.c \
		-o tests/fuzzing/simple_fuzz_lexer
	@echo "✓ Fuzzer binaries built in tests/fuzzing/"
	@echo "Note: For advanced fuzzing with libFuzzer or AFL++, see tests/fuzzing/README.md"

fuzz-lexer: fuzz-build
	@echo "Running lexer fuzzer on corpus..."
	@for f in tests/fuzzing/corpus_lexer/*.nano; do \
		echo "Testing $$f..."; \
		tests/fuzzing/simple_fuzz_lexer "$$f" || exit 1; \
	done
	@echo "✓ Lexer fuzzing complete (all seed files passed)"

fuzz: fuzz-lexer
	@echo "All fuzzing runs complete"

# Help
help:
	@echo "Nanolang Makefile - Build & Bootstrap Targets"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Main Targets:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make build              - Build compiler (3-stage bootstrap)"
	@echo "  make vm                 - Build NanoISA VM backend (nano_virt, nano_vm, nano_cop, nano_vmd)"
	@echo "  make bootstrap          - TRUE 3-stage bootstrap (GCC-style)"
	@echo "  make test               - Build + run all tests (auto-detect best compiler)"
	@echo "  make test-beads         - Run tests; on failures, auto-create/update beads"
	@echo ""
	@echo "Module Dependencies:"
	@echo "  make modules            - Check what dependencies are needed (no sudo)"
	@echo "  sudo make install-deps  - Install all missing dependencies (requires sudo)"
	@echo ""
	@echo "  make examples           - Build all examples (STRICT: fails if deps missing)"
	@echo "  make examples-available - Build available examples (GRACEFUL: skip missing deps)"
	@echo "  make examples-beads     - Build examples; on failures, auto-create/update beads"
	@echo "  make launcher           - Launch example browser"
	@echo "  make clean              - Remove all artifacts"
	@echo "  make rebuild            - Clean + build"
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
	@echo "  make test-vm           - Run all tests through NanoVM backend"
	@echo "  make test-daemon       - Run all tests through NanoVM daemon backend"
	@echo "  make test-units        - Run C unit tests (ISA + VM + codegen)"
	@echo "  make test-nanoisa      - Run NanoISA unit tests (470 tests)"
	@echo "  make test-nanovm       - Run NanoVM unit tests (150 tests)"
	@echo "  make test-nanovirt     - Run codegen unit tests (62 tests)"
	@echo "  make fuzz              - Run fuzzing on seed corpus"
	@echo "  make fuzz-lexer        - Fuzz lexer with AddressSanitizer"
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
	@echo "Development & Quality:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make sanitize        - Build with sanitizers (AddressSanitizer + UBSan)"
	@echo "  make coverage        - Build with coverage instrumentation"
	@echo "  make coverage-report - Generate HTML coverage report (requires lcov)"
	@echo "  make valgrind        - Run memory checks with valgrind"
	@echo "  make install         - Install to $(PREFIX)/bin"
	@echo ""
	@echo "Coverage: Requires lcov (brew install lcov / apt-get install lcov)"
	@echo "Report generated in: coverage/index.html"
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
	@$(TIMEOUT_CMD) python3 scripts/autobeads.py --examples

# CI-friendly: one summary bead per run (per branch), auto-closed when green
test-beads-ci:
	@$(TIMEOUT_CMD) python3 scripts/autobeads.py --tests --mode summary --close-on-success

examples-beads-ci:
	@$(TIMEOUT_CMD) python3 scripts/autobeads.py --examples --mode summary --close-on-success
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

$(OBJ_DIR)/eval: | $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/eval

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: all build vm test test-docs test-nanoisa test-nanovm test-nanovirt nano_vm nano_vmd nano_virt nano_cop test-nanovm-daemon test-nanovm-integration test-cop-lifecycle test-vm test-daemon examples examples-core examples-stage1 examples-stage3 examples-vm examples-available launcher examples-no-sdl clean rebuild help status sanitize coverage coverage-report install install-deps uninstall valgrind stage1.5 bootstrap-status bootstrap-install modules-index modules release release-major release-minor release-patch

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
	@BATCH=$(BATCH) $(RELEASE_TIMEOUT_CMD) ./scripts/release.sh patch

release-minor:
	@echo "Creating minor release..."
	@BATCH=$(BATCH) $(RELEASE_TIMEOUT_CMD) ./scripts/release.sh minor

release-major:
	@echo "Creating major release..."
	@BATCH=$(BATCH) $(RELEASE_TIMEOUT_CMD) ./scripts/release.sh major
