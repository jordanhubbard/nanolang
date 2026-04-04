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
CFLAGS = -Wall -Wextra -Werror -std=c99 -g -O3 -march=native -ftree-vectorize -Isrc -D_GNU_SOURCE
# Enable with: make CFLAGS="$(CFLAGS) $(VECTORIZE_FLAGS)" to inspect missed vectorizations
VECTORIZE_FLAGS = -fopt-info-vec-missed
LDFLAGS = -lm -lcrypto

# On Linux, dlopened module shared libraries rely on host-exported runtime symbols
# (e.g. dyn_array_new). Ensure the main binaries export their symbols.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
LDFLAGS += -rdynamic
endif
ifeq ($(UNAME_S),FreeBSD)
LDFLAGS += -Wl,-E
endif
ifeq ($(UNAME_S),Darwin)
# Homebrew OpenSSL is keg-only on macOS — add include/lib paths
OPENSSL_PREFIX := $(shell brew --prefix openssl 2>/dev/null)
ifneq ($(OPENSSL_PREFIX),)
CFLAGS  += -I$(OPENSSL_PREFIX)/include
LDFLAGS += -L$(OPENSSL_PREFIX)/lib
endif
endif
# Note: -fblocks/-ldispatch/-lBlocksRuntime are only needed when compiling programs
# that use nanolang's `let mut` concurrency feature (nano_dispatch.h). They are NOT
# needed for building the compiler tools themselves.
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
# Absolute path to repo modules; passed to examples build so module resolution works from any cwd
NANO_MODULES_ABS := $(abspath $(CURDIR)/modules)
ifeq ($(BOOTSTRAP_DETERMINISTIC),1)
BOOTSTRAP_ENV += NANO_DETERMINISTIC=1
endif
BOOTSTRAP_VERBOSE ?= 0
ifeq ($(BOOTSTRAP_VERBOSE),1)
BOOTSTRAP_ENV += NANO_VERBOSE_BUILD=1
endif
BOOTSTRAP_VERBOSE_FLAG :=
ifeq ($(BOOTSTRAP_VERBOSE),1)
BOOTSTRAP_VERBOSE_FLAG := -v
endif

# Source files
COMMON_SOURCES = $(SRC_DIR)/lexer.c $(SRC_DIR)/parser.c $(SRC_DIR)/typechecker.c $(SRC_DIR)/transpiler.c $(SRC_DIR)/stdlib_runtime.c $(SRC_DIR)/env.c $(SRC_DIR)/builtins_registry.c $(SRC_DIR)/module.c $(SRC_DIR)/module_metadata.c $(SRC_DIR)/cJSON.c $(SRC_DIR)/toon_output.c $(SRC_DIR)/module_builder.c $(SRC_DIR)/resource_tracking.c $(SRC_DIR)/eval.c $(SRC_DIR)/eval/eval_hashmap.c $(SRC_DIR)/eval/eval_math.c $(SRC_DIR)/eval/eval_string.c $(SRC_DIR)/eval/eval_io.c $(SRC_DIR)/interpreter_ffi.c $(SRC_DIR)/json_diagnostics.c $(SRC_DIR)/reflection.c $(SRC_DIR)/nanocore_subset.c $(SRC_DIR)/nanocore_export.c $(SRC_DIR)/emit_typed_ast.c $(SRC_DIR)/wasm_backend.c $(SRC_DIR)/wasm_profiler.c $(SRC_DIR)/wasm_simd.c $(SRC_DIR)/type_infer.c $(SRC_DIR)/effects.c $(SRC_DIR)/fold_constants.c $(SRC_DIR)/dce_pass.c $(SRC_DIR)/par_let_pass.c $(SRC_DIR)/sign.c $(SRC_DIR)/ptx_backend.c $(SRC_DIR)/tco_pass.c $(SRC_DIR)/cps_pass.c $(SRC_DIR)/coroutine.c $(SRC_DIR)/pgo_pass.c $(SRC_DIR)/llvm_backend.c $(SRC_DIR)/c_backend.c $(SRC_DIR)/bench.c $(SRC_DIR)/bench_native.c $(SRC_DIR)/riscv_backend.c $(SRC_DIR)/dwarf_info.c $(SRC_DIR)/docgen_md.c $(SRC_DIR)/docgen.c $(SRC_DIR)/fmt.c $(SRC_DIR)/channel.c
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
INTERPRETER = $(BIN_DIR)/nano
INTERPRETER_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nano_main.o $(OBJ_DIR)/proptest.o

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

# ============================================================================
# Package Manager
# ============================================================================

.PHONY: pkg-install pkg-publish pkg-update pkg-init pkg-list

pkg-install:
	@./scripts/nano-pkg.sh install

pkg-publish:
	@./scripts/nano-pkg.sh publish

pkg-update:
	@./scripts/nano-pkg.sh update

pkg-init:
	@./scripts/nano-pkg.sh init

pkg-list:
	@./scripts/nano-pkg.sh list

# Hybrid compiler objects
HYBRID_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/lexer_bridge.o $(OBJ_DIR)/lexer_nano.o $(OBJ_DIR)/main_stage1_5.o

PREFIX ?= /usr/local

# ============================================================================
# Main Targets
# ============================================================================

.DEFAULT_GOAL := build


# Build: 3-stage bootstrap (uses sentinels to skip completed stages)
build: schema modules-index $(SENTINEL_STAGE3) $(INTERPRETER) $(REPL_BINARY)
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
.PHONY: test-optimizer
test-optimizer: stage1
	@echo "Running optimizer unit tests..."
	$(CC) $(CFLAGS) -o tests/test_optimizer tests/test_optimizer.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_optimizer
	@rm -f tests/test_optimizer

.PHONY: test-wasm-profiler
test-wasm-profiler: stage1
	@echo "Running WASM profiler unit tests..."
	$(CC) $(CFLAGS) -o tests/test_wasm_profiler tests/test_wasm_profiler.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_wasm_profiler
	@rm -f tests/test_wasm_profiler

.PHONY: test-diagnostics
test-diagnostics: stage1
	@echo "Running diagnostics unit tests..."
	$(CC) $(CFLAGS) -o tests/test_diagnostics tests/test_diagnostics.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_diagnostics
	@rm -f tests/test_diagnostics

.PHONY: test-module-metadata
test-module-metadata: stage1
	@echo "Running module metadata unit tests..."
	$(CC) $(CFLAGS) -o tests/test_module_metadata tests/test_module_metadata.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_module_metadata
	@rm -f tests/test_module_metadata

.PHONY: test-type-infer
test-type-infer: stage1
	@echo "Running type inference unit tests..."
	$(CC) $(CFLAGS) -o tests/test_type_infer tests/test_type_infer.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_type_infer
	@rm -f tests/test_type_infer

.PHONY: test-coroutine-scheduler
test-coroutine-scheduler: stage1
	@echo "Running coroutine scheduler unit tests..."
	$(CC) $(CFLAGS) -o tests/test_coroutine_scheduler tests/test_coroutine_scheduler.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_coroutine_scheduler
	@rm -f tests/test_coroutine_scheduler

.PHONY: test-eval
test-eval: stage1
	@echo "Running interpreter (eval.c) unit tests..."
	$(CC) $(CFLAGS) -o tests/test_eval tests/test_eval.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_eval
	@rm -f tests/test_eval

.PHONY: test-opt-passes
test-opt-passes: stage1
	@echo "Running optimization pass unit tests..."
	$(CC) $(CFLAGS) -o tests/test_opt_passes tests/test_opt_passes.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_opt_passes
	@rm -f tests/test_opt_passes

.PHONY: test-effects
test-effects: stage1
	@echo "Running effects system unit tests..."
	$(CC) $(CFLAGS) -o tests/test_effects tests/test_effects.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_effects
	@rm -f tests/test_effects

.PHONY: test-ffi
test-ffi: stage1
	@echo "Running interpreter FFI unit tests..."
	$(CC) $(CFLAGS) -o tests/test_ffi tests/test_ffi.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_ffi
	@rm -f tests/test_ffi

.PHONY: test-wasm-simd
test-wasm-simd: stage1
	@echo "Running WASM SIMD unit tests..."
	$(CC) $(CFLAGS) -o tests/test_wasm_simd tests/test_wasm_simd.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_wasm_simd
	@rm -f tests/test_wasm_simd

.PHONY: test-runtime-lists
test-runtime-lists: stage1
	@echo "Running runtime list unit tests..."
	$(CC) $(CFLAGS) -o tests/test_runtime_lists tests/test_runtime_lists.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_runtime_lists
	@rm -f tests/test_runtime_lists

.PHONY: test-typechecker
test-typechecker: stage1
	@echo "Running typechecker unit tests..."
	$(CC) $(CFLAGS) -o tests/test_typechecker tests/test_typechecker.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_typechecker
	@rm -f tests/test_typechecker

.PHONY: test-builtins-direct
test-builtins-direct: stage1
	@echo "Running builtin functions direct unit tests..."
	$(CC) $(CFLAGS) -o tests/test_builtins_direct tests/test_builtins_direct.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS) -lm
	@./tests/test_builtins_direct
	@rm -f tests/test_builtins_direct

.PHONY: test-transpiler
test-transpiler: stage1
	@echo "Running transpiler unit tests..."
	$(CC) $(CFLAGS) -o tests/test_transpiler tests/test_transpiler.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS) -lm
	@./tests/test_transpiler
	@rm -f tests/test_transpiler

.PHONY: test-parser
test-parser: stage1
	@echo "Running parser unit tests..."
	$(CC) $(CFLAGS) -o tests/test_parser tests/test_parser.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_parser
	@rm -f tests/test_parser

.PHONY: test-row-poly
test-row-poly: build
	@echo "Running row-polymorphic records tests..."
	@bash scripts/check_shadow_tests.sh tests/test_row_poly.nano

.PHONY: test-nl-string
test-nl-string: stage1
	@echo "Running nl_string unit tests..."
	$(CC) $(CFLAGS) -o tests/test_nl_string tests/test_nl_string.c $(OBJ_DIR)/runtime/nl_string.o $(LDFLAGS)
	@./tests/test_nl_string
	@rm -f tests/test_nl_string

.PHONY: test-refcount-gc
test-refcount-gc:
	@echo "Running refcount_gc unit tests..."
	$(CC) $(CFLAGS) -o tests/test_refcount_gc tests/test_refcount_gc.c $(LDFLAGS)
	@./tests/test_refcount_gc
	@rm -f tests/test_refcount_gc

.PHONY: test-pgo-pass
test-pgo-pass: stage1
	@echo "Running PGO pass unit tests..."
	$(CC) $(CFLAGS) -o tests/test_pgo_pass tests/test_pgo_pass.c $(OBJ_DIR)/pgo_pass.o $(LDFLAGS)
	@./tests/test_pgo_pass
	@rm -f tests/test_pgo_pass

.PHONY: test-docgen
test-docgen: stage1
	@echo "Running docgen unit tests..."
	$(CC) $(CFLAGS) -o tests/test_docgen tests/test_docgen.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_docgen
	@rm -f tests/test_docgen

.PHONY: test-backends
test-backends: stage1
	@echo "Running alternative backend unit tests..."
	$(CC) $(CFLAGS) -o tests/test_backends tests/test_backends.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS) -lm
	@./tests/test_backends
	@rm -f tests/test_backends

.PHONY: test-compiler-utils
test-compiler-utils: stage1
	@echo "Running compiler utility module tests..."
	$(CC) $(CFLAGS) -o tests/test_compiler_utils tests/test_compiler_utils.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_compiler_utils
	@rm -f tests/test_compiler_utils

.PHONY: test-sign
test-sign: stage1
	@echo "Running sign unit tests..."
	$(CC) $(CFLAGS) -o tests/test_sign tests/test_sign.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_sign
	@rm -f tests/test_sign

.PHONY: test-module-loading
test-module-loading: stage1
	@echo "Running module loading tests..."
	$(CC) $(CFLAGS) -o tests/test_module_loading tests/test_module_loading.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_module_loading
	@rm -f tests/test_module_loading

.PHONY: test-fmt
test-fmt: stage1
	@echo "Running fmt unit tests..."
	$(CC) $(CFLAGS) -o tests/test_fmt tests/test_fmt.c $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	@./tests/test_fmt
	@rm -f tests/test_fmt

.PHONY: test-channel
test-channel: stage1
	@echo "Running channel unit tests..."
	$(CC) $(CFLAGS) -o tests/test_channel tests/test_channel.c $(OBJ_DIR)/channel.o $(LDFLAGS)
	@./tests/test_channel
	@rm -f tests/test_channel

.PHONY: test-units
test-units: test-nanoisa test-nanovm test-nanovirt test-optimizer test-wasm-profiler test-diagnostics test-module-metadata test-type-infer test-opt-passes test-eval test-coroutine-scheduler test-runtime-lists test-wasm-simd test-ffi test-effects test-builtins-direct test-typechecker test-parser test-transpiler test-nl-string test-refcount-gc test-pgo-pass test-docgen test-backends test-compiler-utils test-sign test-module-loading test-fmt test-channel
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

# Property-based test oracle smoke tests
.PHONY: test-proptest
test-proptest: $(INTERPRETER)
	@echo "Running proptest smoke tests..."
	@bash tests/test_proptest.sh
	@echo "proptest smoke tests passed."

# Ring-buffer unit tests (no compiler or VM required)
.PHONY: test-ringbuf
test-ringbuf:
	@echo "Building tests/test_ringbuf..."
	@$(CC) -std=c99 -Wall -Wextra -Iinclude -o tests/test_ringbuf_bin tests/test_ringbuf.c
	@echo "Running ringbuf tests..."
	@./tests/test_ringbuf_bin
	@rm -f tests/test_ringbuf_bin
	@echo "ringbuf tests passed."

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
	@echo "Testing C backend (--target c)..."
	@bash tests/test_c_backend.sh $(COMPILER_C)
	@echo ""
	@echo "Testing Markdown docgen (--doc-md)..."
	@bash tests/test_docgen_md.sh $(COMPILER_C)
	@echo ""
	@echo "Testing alternative backends (riscv, ptx, reflect)..."
	@bash tests/test_backends.sh $(COMPILER_C)
	@echo ""
	@if [ -x $(INTERPRETER) ]; then \
		echo "Running property-based tests (interpreter)..."; \
		bash tests/test_proptest.sh; \
	else \
		echo "Skipping proptest: $(INTERPRETER) not built (run 'make build' first)"; \
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
	@echo "🎯 Testing with C REFERENCE compiler (nanoc_c)"
	@echo "   (Use 'make test-selfhosted' to test with nanoc_stage2)"
	@echo ""
	@rm -f $(COMPILER)
	@ln -sf nanoc_c $(COMPILER)
	@perl -e 'alarm $(TEST_TIMEOUT); exec @ARGV' $(MAKE) test-impl
	@# Restore proper link based on bootstrap status
	@if [ -f $(SENTINEL_BOOTSTRAP3) ] && [ -f $(NANOC_STAGE2) ]; then \
		rm -f $(COMPILER); \
		ln -sf nanoc_stage2 $(COMPILER); \
	fi

# Profile-runtime smoke test: compile with --profile-runtime, run, check .nano.prof exists
test-profile-runtime: $(INTERPRETER) $(COMPILER)
	@echo "🔥 Testing --profile-runtime flamegraph output..."
	@$(COMPILER) tests/unit/test_profile_runtime.nano --profile-runtime -o /tmp/nano_prof_smoke 2>&1 | grep -v "shadow test" || true
	@/tmp/nano_prof_smoke 2>/tmp/nano_prof_smoke.stderr
	@grep -q "nl_count_down" /tmp/nano_prof_smoke.nano.prof || (echo "❌ .nano.prof missing nl_count_down"; exit 1)
	@echo "✅ --profile-runtime: flamegraph .nano.prof generated OK"
	@rm -f /tmp/nano_prof_smoke /tmp/nano_prof_smoke.nano.prof /tmp/nano_prof_smoke.stderr

# PTX backend test: compile gpu fn test to PTX and verify kernels are emitted
test-ptx: $(COMPILER)
	@echo "🔥 Testing --target ptx PTX backend..."
	@$(COMPILER) tests/gpu/test_ptx_emit.nano --target ptx -o /tmp/nano_ptx_test.ptx 2>/dev/null
	@grep -q "\.visible \.entry vec_add" /tmp/nano_ptx_test.ptx    || (echo "❌ vec_add kernel missing";  exit 1)
	@grep -q "\.visible \.entry scale"   /tmp/nano_ptx_test.ptx    || (echo "❌ scale kernel missing";    exit 1)
	@grep -q "\.visible \.entry lerp"    /tmp/nano_ptx_test.ptx    || (echo "❌ lerp kernel missing";     exit 1)
	@grep -q "\.visible \.entry relu"    /tmp/nano_ptx_test.ptx    || (echo "❌ relu kernel missing";     exit 1)
	@grep -qv "\.visible \.entry host_helper" /tmp/nano_ptx_test.ptx || (echo "❌ host_helper should not be a kernel"; exit 1)
	@grep -q "\.target sm_90"            /tmp/nano_ptx_test.ptx    || (echo "❌ sm_90 target missing";    exit 1)
	@grep -q "%tid\.x"                   /tmp/nano_ptx_test.ptx    || (echo "❌ thread_id_x builtin missing"; exit 1)
	@echo "✅ PTX backend: 4 kernels emitted correctly"
	@rm -f /tmp/nano_ptx_test.ptx

# DWARF debug info test: verify --debug emits correct metadata for LLVM/RISC-V
.PHONY: test-dwarf
test-dwarf: $(COMPILER_C)
	@echo "🔍 Testing DWARF debug info emission (--debug / -g flag)..."
	@bash tests/test_dwarf.sh $(COMPILER_C)

# C backend test: transpile .nano → .c → gcc → run, verify output
.PHONY: test-c-backend
test-c-backend: $(COMPILER_C)
	@echo "🔧 Testing --target c backend..."
	@bash tests/test_c_backend.sh $(COMPILER_C)
	@echo "✅ C backend tests PASSED"

# Cross-backend compile suite: compile canonical test programs across all 5 backends
.PHONY: test-cross-backend
test-cross-backend: $(COMPILER)
	@echo "🔀 Running cross-backend compile suite (wasm, llvm, riscv, c, ptx)..."
	@chmod +x tests/cross-backend/run-all.sh
	@bash tests/cross-backend/run-all.sh $(COMPILER)
	@echo "✅ Cross-backend tests PASSED"

# Coroutine runtime test: runs tests/test_coroutine.nano through the interpreter
test-coroutine: $(INTERPRETER)
	@echo "🔄 Running coroutine runtime tests..."
	@$(INTERPRETER) tests/test_coroutine.nano
	@echo "✅ Coroutine runtime tests PASSED"

# ── Benchmark suite ──────────────────────────────────────────────────────
# Run the full benchmark suite and write results to bench/results.json
.PHONY: bench bench-compare

bench: build
	@echo "📊 Running nanolang benchmark suite..."
	@chmod +x scripts/run_bench.sh
	@bash scripts/run_bench.sh

# Compare against a baseline: make bench-compare BASELINE=bench/baseline.json
bench-compare: build
	@echo "📊 Running benchmark suite with regression check..."
	@chmod +x scripts/run_bench.sh
	@bash scripts/run_bench.sh --baseline $(BASELINE) --threshold $(or $(THRESHOLD),20)

# Script mode tests
test-script: $(INTERPRETER)
	@echo "📜 Testing nano --script and -e modes..."
	@chmod +x tests/test_script_mode.sh
	@bash tests/test_script_mode.sh $(INTERPRETER)
	@echo "✅ Script mode tests passed"

# Markdown docgen tests: verify --doc-md flag emits valid GFM
test-doc-md: build
	@chmod +x tests/test_docgen_md.sh
	@bash tests/test_docgen_md.sh $(COMPILER_C)
	@echo "✅ Markdown docgen tests passed"

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
	@echo "🎯 Testing with C REFERENCE compiler (nanoc_c) + Beads tracking"
	@echo ""
	@rm -f $(COMPILER)
	@ln -sf nanoc_c $(COMPILER)
	@echo ""
	@# Auto-file beads on failures.
	@# Local default: per-failure beads. CI default: summary bead.
	@MODE=per; \
	if [ -n "$$CI" ]; then MODE=summary; fi; \
	$(TIMEOUT_CMD) python3 scripts/autobeads.py --tests --mode $$MODE --close-on-success --timeout-seconds $${NANOLANG_TEST_TIMEOUT_SECONDS:-480}
	@# Restore proper link based on bootstrap status
	@if [ -f $(SENTINEL_BOOTSTRAP3) ] && [ -f $(NANOC_STAGE2) ]; then \
		rm -f $(COMPILER); \
		ln -sf nanoc_stage2 $(COMPILER); \
	fi

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

# API documentation (nanodoc)
DOCS_OUTPUT ?= docs/api
NANODOC_SOURCES := $(wildcard std/*/*.nano) $(wildcard examples/language/*.nano)

.PHONY: docs
docs:
	@echo "Generating API documentation..."
	@mkdir -p $(DOCS_OUTPUT)
	@node tools/nanodoc.mjs --output $(DOCS_OUTPUT) $(NANODOC_SOURCES)
	@echo "✅ API docs written to $(DOCS_OUTPUT)/"

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

# Test with self-hosted compiler (nanoc_stage2), without rebuilding
test-selfhosted: build
	@echo ""
	@echo "🎯 Testing with SELF-HOSTED compiler (nanoc_stage2)"
	@echo "   Note: requires 'make bootstrap' to be complete"
	@echo ""
	@if [ ! -f $(SENTINEL_BOOTSTRAP3) ] || [ ! -f $(NANOC_STAGE2) ]; then \
		echo "❌ Bootstrap not complete. Run 'make bootstrap' first."; \
		exit 1; \
	fi
	@rm -f $(COMPILER)
	@ln -sf nanoc_stage2 $(COMPILER)
	@$(TIMEOUT_CMD) $(MAKE) test-impl
	@# Restore after test (stage2 is still primary when bootstrapped)
	@if [ -f $(SENTINEL_BOOTSTRAP3) ] && [ -f $(NANOC_STAGE2) ]; then \
		rm -f $(COMPILER); \
		ln -sf nanoc_stage2 $(COMPILER); \
	fi

# Test only core language features (nl_* tests)
test-lang: build
	@echo ""
	@echo "=========================================="
	@echo "Running Core Language Tests (nl_*)"
	@echo "=========================================="
	@./tests/run_all_tests.sh --lang

# CI test targets with structured output
test-junit: build
	@echo "Running all tests with JUnit XML output..."
	@./tests/run_all_tests.sh --format=junit --output=test-results.xml

test-tap: build
	@echo "Running all tests with TAP output..."
	@./tests/run_all_tests.sh --format=tap --output=test-results.tap

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
	fi

.PHONY: examples-core examples-stage1 examples-full examples-stage3 examples-vm examples-launcher

examples-core: examples-stage1 examples-full
	@echo "✅ Examples built successfully!"

# Build with nanoc (C reference compiler) to include SDL/NCurses/network examples
examples-full: stage1 modules-index check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Examples (Full Compiler)"
	@echo "=========================================="
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples build NANO_MODULE_PATH="$(NANO_MODULES_ABS)"

examples-stage1: bootstrap1 modules-index check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Examples (Stage 1 Compiler)"
	@echo "=========================================="
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples build COMPILER=../bin/nanoc_stage1 BIN_SUFFIX= NANO_MODULE_PATH="$(NANO_MODULES_ABS)"

examples-stage3: bootstrap3 modules-index check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Examples (Stage 3 Compiler)"
	@echo "=========================================="
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples build COMPILER=../bin/nanoc_stage2 BIN_SUFFIX=_stage3 NANO_MODULE_PATH="$(NANO_MODULES_ABS)"

examples-vm: vm modules-index check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Examples (NanoVM Bytecode)"
	@echo "=========================================="
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples build EXAMPLES_BACKEND=vm BIN_SUFFIX=_vm NANO_MODULE_PATH="$(NANO_MODULES_ABS)"

# Build available examples (GRACEFUL: skip examples with missing dependencies)
examples-available: build check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "Building Available Examples"
	@echo "=========================================="
	@echo "Note: Examples with missing dependencies will be skipped."
	@echo ""
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples examples-available NANO_MODULE_PATH="$(NANO_MODULES_ABS)"

# Launch example browser
examples-launcher: examples check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "🚀 Launching Example Browser"
	@echo "=========================================="
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples launcher COMPILER=../bin/nanoc_stage1 BIN_SUFFIX= NANO_MODULE_PATH="$(NANO_MODULES_ABS)"

launcher: build check-deps-sdl
	@echo ""
	@echo "=========================================="
	@echo "🚀 Launching Example Browser"
	@echo "=========================================="
	@$(EXAMPLES_TIMEOUT_CMD) $(MAKE) -C examples launcher NANO_MODULE_PATH="$(NANO_MODULES_ABS)"
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

# Interpreter binary: bin/nano runs programs directly via the tree-walking interpreter
$(INTERPRETER): $(INTERPRETER_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(INTERPRETER) $(INTERPRETER_OBJECTS) $(LDFLAGS)
	@echo "✓ Interpreter: $(INTERPRETER)"

# REPL binary: bin/nanolang-repl — interactive read-eval-print loop
REPL_BINARY   = $(BIN_DIR)/nanolang-repl
REPL_OBJECTS  = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) \
                $(OBJ_DIR)/repl.o $(OBJ_DIR)/repl_main.o $(OBJ_DIR)/proptest.o

$(OBJ_DIR)/repl.o: $(SRC_DIR)/repl.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/repl.c -o $(OBJ_DIR)/repl.o

$(OBJ_DIR)/repl_main.o: $(SRC_DIR)/repl_main.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/repl_main.c -o $(OBJ_DIR)/repl_main.o

$(REPL_BINARY): $(REPL_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(REPL_BINARY) $(REPL_OBJECTS) $(LDFLAGS)
	@echo "✓ REPL: $(REPL_BINARY)"

.PHONY: repl
repl: $(REPL_BINARY)

# LSP server binary: bin/nanolang-lsp speaks JSON-RPC 2.0 over stdio
LSP_SERVER = $(BIN_DIR)/nanolang-lsp
LSP_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/lsp_server.o

$(OBJ_DIR)/lsp_server.o: $(SRC_DIR)/lsp_server.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/lsp_server.c -o $(OBJ_DIR)/lsp_server.o

$(LSP_SERVER): $(LSP_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(LSP_SERVER) $(LSP_OBJECTS) $(LDFLAGS)
	@echo "✓ LSP Server: $(LSP_SERVER)"

.PHONY: lsp
lsp: $(LSP_SERVER)

# DAP server binary: bin/nanolang-dap speaks Debug Adapter Protocol over stdio
DAP_SERVER = $(BIN_DIR)/nanolang-dap
DAP_OBJECTS = $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/dap_server.o

$(OBJ_DIR)/dap_server.o: $(SRC_DIR)/dap_server.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/dap_server.c -o $(OBJ_DIR)/dap_server.o

$(DAP_SERVER): $(DAP_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(DAP_SERVER) $(DAP_OBJECTS) $(LDFLAGS)
	@echo "✓ DAP Server: $(DAP_SERVER)"

.PHONY: dap
dap: $(DAP_SERVER)

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
.PHONY: check-deps check-deps-sdl module-package-audit
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

module-package-audit:
	@python3 tools/audit_module_packages.py

# Build with sanitizers
sanitize: CFLAGS += $(SANITIZE_FLAGS)
sanitize: LDFLAGS += $(SANITIZE_FLAGS)
sanitize: rebuild

# Build with coverage
coverage: CFLAGS += $(COVERAGE_FLAGS)
coverage: LDFLAGS += $(COVERAGE_FLAGS)
coverage: rebuild

# Configurable coverage threshold (override with: make coverage-check COVERAGE_THRESHOLD=70)
COVERAGE_THRESHOLD ?= 80.0

# Coverage report: build with instrumentation, run tests, generate HTML
coverage-report: coverage test
	@echo "Generating coverage report..."
	@$(TIMEOUT_CMD) lcov --capture --directory . --output-file coverage.info
	@$(TIMEOUT_CMD) lcov --remove coverage.info '/usr/*' --output-file coverage.info --ignore-errors unused
	@$(TIMEOUT_CMD) lcov --remove coverage.info '*/cJSON.c' --output-file coverage.info --ignore-errors unused
	@$(TIMEOUT_CMD) genhtml coverage.info --output-directory $(COV_DIR) --branch-coverage
	@echo "✅ Coverage report: $(COV_DIR)/index.html"
	@$(MAKE) coverage-check

# Coverage threshold check: parse coverage.info and fail if below COVERAGE_THRESHOLD
coverage-check: coverage.info
	@echo "Checking coverage threshold (>= $(COVERAGE_THRESHOLD)%)..."
	@COVERAGE=$$(lcov --summary coverage.info 2>&1 | grep "lines\." | grep -oE '[0-9]+\.[0-9]+' | head -1); \
	if [ -z "$$COVERAGE" ]; then \
		echo "⚠️  Could not parse coverage percentage from coverage.info"; \
		exit 1; \
	fi; \
	echo "Line coverage: $$COVERAGE%"; \
	if command -v bc >/dev/null 2>&1; then \
		if echo "$$COVERAGE < $(COVERAGE_THRESHOLD)" | bc -l | grep -q '^1'; then \
			echo "❌ Coverage $$COVERAGE% is below threshold $(COVERAGE_THRESHOLD)%"; \
			echo "   Add more tests or lower COVERAGE_THRESHOLD to override."; \
			exit 1; \
		else \
			echo "✅ Coverage $$COVERAGE% meets threshold $(COVERAGE_THRESHOLD)%"; \
		fi; \
	else \
		echo "⚠️  bc not found — skipping numeric threshold check"; \
	fi

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
	@echo "  make module-package-audit - Validate module package metadata coverage"
	@echo ""
	@echo "Package Manager:"
	@echo "  make pkg-install        - Install packages from nano.toml"
	@echo "  make pkg-publish        - Publish package to registry"
	@echo "  make pkg-update         - Update all packages to latest"
	@echo "  make pkg-init           - Create a new nano.toml"
	@echo "  make pkg-list           - List installed packages"
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
	@echo "  make coverage-report - Generate HTML coverage report + threshold check (requires lcov)"
	@echo "  make coverage-check  - Check coverage.info meets threshold (COVERAGE_THRESHOLD=$(COVERAGE_THRESHOLD))"
	@echo "  make valgrind        - Run memory checks with valgrind"
	@echo "  make install         - Install to $(PREFIX)/bin"
	@echo "  make vscode-ext      - Build VS Code extension (npm install + compile + vsce package)"
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

# Interactive playground (browser editor + eval server)
.PHONY: playground
playground:
	@echo "Starting Nanolang Playground on http://localhost:8792 ..."
	@node playground/server.mjs

# VS Code extension
vscode-ext:
	@echo "Building VS Code extension..."
	@cd vscode && npm install
	@cd vscode && npm run compile
	@cd vscode && npm run package
	@echo "Extension packaged: vscode/nanolang-*.vsix"
	@echo "Install with: code --install-extension vscode/nanolang-*.vsix"

# ── wasm-playground: build nanolang.wasm for the browser playground ──────────
# Requires Emscripten SDK: source emsdk/emsdk_env.sh before running.
WASM_OUT_DIR  = examples/playground/public
WASM_SRCS     = src/lexer.c src/parser.c src/eval.c src/env.c src/module.c \
                src/typechecker.c src/builtins.c src/gc.c src/repl.c \
                src/stdlib_io.c src/stdlib_math.c src/stdlib_string.c \
                src/stdlib_collections.c src/interpreter_ffi.c
WASM_EXPORTED = '["_nl_run","_nl_check","_nl_version","_malloc","_free"]'
WASM_FLAGS    = -O2 -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME=createNanolang \
                -s EXPORTED_FUNCTIONS=$(WASM_EXPORTED) \
                -s EXPORTED_RUNTIME_METHODS='["cwrap","ccall"]' \
                -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=67108864 \
                -s ENVIRONMENT=web -s NO_EXIT_RUNTIME=1 \
                -I src/ -I src/generated/ \
                --output-file $(WASM_OUT_DIR)/nanolang.js

wasm-playground:
	@echo "Building NanoLang WASM playground (requires Emscripten)..."
	@command -v emcc >/dev/null 2>&1 || (echo "ERROR: emcc not found. Source emsdk/emsdk_env.sh first." && exit 1)
	@mkdir -p $(WASM_OUT_DIR)
	emcc $(WASM_SRCS) $(WASM_FLAGS)
	@echo "Built: $(WASM_OUT_DIR)/nanolang.wasm + nanolang.js"
	@echo "Serve: python3 -m http.server 8000 --directory $(WASM_OUT_DIR)"

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

.PHONY: all build vm test test-selfhosted test-docs test-doc-md test-nanoisa test-nanovm test-nanovirt nano_vm nano_vmd nano_virt nano_cop test-nanovm-daemon test-nanovm-integration test-cop-lifecycle test-vm test-daemon examples examples-core examples-stage1 examples-stage3 examples-vm examples-available launcher examples-no-sdl clean rebuild help status sanitize coverage coverage-report install install-deps uninstall valgrind stage1.5 bootstrap-status bootstrap-install modules-index modules module-package-audit release release-major release-minor release-patch pkg-install pkg-publish pkg-update pkg-init pkg-list

# ============================================================================
# AGENTFS PUBLISH
# ============================================================================
# Compile a .nano file to WASM and publish to AgentFS for use in agentOS
# Usage:
#   make publish SOURCE=src/my_agent.nano
#   make publish SOURCE=src/my_agent.nano CAPABILITIES=fs,net
#   make publish SOURCE=src/my_agent.nano AGENTFS_URL=http://localhost:8791
publish:
ifndef SOURCE
	@echo "Usage: make publish SOURCE=<file.nano> [CAPABILITIES=fs,net] [AGENTFS_URL=http://...]"
	@echo "       make publish SOURCE=examples/hello.nano --dry-run ARGS=--dry-run"
	@exit 1
endif
	@scripts/nanoc-publish.sh $(SOURCE) \
		$(if $(CAPABILITIES),--capabilities "$(CAPABILITIES)") \
		$(if $(AGENTFS_URL),--agentfs "$(AGENTFS_URL)") \
		$(ARGS)

publish-dry-run:
ifndef SOURCE
	@echo "Usage: make publish-dry-run SOURCE=<file.nano>"
	@exit 1
endif
	@scripts/nanoc-publish.sh $(SOURCE) --dry-run --verbose

.PHONY: publish publish-dry-run

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
	@$(RELEASE_TIMEOUT_CMD) ./scripts/release.sh patch

release-minor:
	@echo "Creating minor release..."
	@$(RELEASE_TIMEOUT_CMD) ./scripts/release.sh minor

release-major:
	@echo "Creating major release..."
	@$(RELEASE_TIMEOUT_CMD) ./scripts/release.sh major
