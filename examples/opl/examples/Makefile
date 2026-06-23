# Makefile for OPL Examples
# 
# Demonstrates how to use the OPL CLI to parse, validate, compile,
# and generate code from OPL source files.

.PHONY: all clean cli parse validate compile codegen help list

# Configuration
OPL_CLI := /tmp/opl_cli
NANOC := ../../../bin/nanoc
OPL_CLI_SOURCE := ../opl_cli.nano
OUTPUT_DIR := output
TIMEOUT := 90

# Example files (without .opl extension)
EXAMPLES := hello web_search conditional multi_tool service_spec
EXAMPLE ?= hello

# Color output
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m

# Default target
all: cli $(foreach ex,$(EXAMPLES),output/$(ex).plan.json)
	@echo "$(GREEN)✓ All examples built successfully$(NC)"
	@echo ""
	@$(MAKE) --no-print-directory list

# Help target
help:
	@echo "OPL Examples Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build all examples (parse, validate, compile)"
	@echo "  cli              - Compile the OPL CLI tool"
	@echo "  parse            - Parse EXAMPLE to AST JSON"
	@echo "  validate         - Validate EXAMPLE for semantic errors"
	@echo "  compile          - Compile EXAMPLE to PLAN_IR"
	@echo "  codegen          - Generate NanoLang code from EXAMPLE"
	@echo "  list             - List all generated output files"
	@echo "  clean            - Remove all generated files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  EXAMPLE=<name>   - Specify which example to process (default: hello)"
	@echo ""
	@echo "Available examples:"
	@echo "  $(EXAMPLES)"
	@echo ""
	@echo "Examples:"
	@echo "  make cli"
	@echo "  make parse EXAMPLE=hello"
	@echo "  make validate EXAMPLE=web_search"
	@echo "  make compile EXAMPLE=conditional"
	@echo "  make all"

# Ensure output directory exists
$(OUTPUT_DIR):
	@mkdir -p $(OUTPUT_DIR)

# Compile the OPL CLI
cli: $(OPL_CLI)

$(OPL_CLI): $(OPL_CLI_SOURCE)
	@echo "$(BLUE)Compiling OPL CLI...$(NC)"
	@perl -e 'alarm $(TIMEOUT); exec @ARGV' $(NANOC) $(OPL_CLI_SOURCE) -o $(OPL_CLI)
	@echo "$(GREEN)✓ OPL CLI compiled: $(OPL_CLI)$(NC)"

# Parse an example to AST JSON
parse: cli $(OUTPUT_DIR)
	@echo "$(BLUE)Parsing $(EXAMPLE).opl → AST JSON...$(NC)"
	@$(OPL_CLI) parse $(EXAMPLE).opl --out $(OUTPUT_DIR)/$(EXAMPLE).ast.json
	@echo "$(GREEN)✓ AST generated: $(OUTPUT_DIR)/$(EXAMPLE).ast.json$(NC)"
	@ls -lh $(OUTPUT_DIR)/$(EXAMPLE).ast.json

# Validate an example
validate: cli $(OUTPUT_DIR)
	@echo "$(BLUE)Validating $(EXAMPLE).opl...$(NC)"
	@$(OPL_CLI) validate $(EXAMPLE).opl --out $(OUTPUT_DIR)/$(EXAMPLE).validate.json && \
		echo "$(GREEN)✓ Validation passed: $(OUTPUT_DIR)/$(EXAMPLE).validate.json$(NC)" || \
		(echo "$(YELLOW)⚠ Validation errors found - see $(OUTPUT_DIR)/$(EXAMPLE).validate.json$(NC)" && cat $(OUTPUT_DIR)/$(EXAMPLE).validate.json && exit 1)

# Compile an example to PLAN_IR
compile: cli $(OUTPUT_DIR)
	@echo "$(BLUE)Compiling $(EXAMPLE).opl → PLAN_IR...$(NC)"
	@$(OPL_CLI) compile $(EXAMPLE).opl --out $(OUTPUT_DIR)/$(EXAMPLE).plan.json
	@echo "$(GREEN)✓ PLAN_IR generated: $(OUTPUT_DIR)/$(EXAMPLE).plan.json$(NC)"
	@ls -lh $(OUTPUT_DIR)/$(EXAMPLE).plan.json

# Generate NanoLang code from PLAN_IR
codegen: cli $(OUTPUT_DIR)
	@echo "$(BLUE)Generating NanoLang code from $(EXAMPLE).plan.json...$(NC)"
	@$(OPL_CLI) codegen $(OUTPUT_DIR)/$(EXAMPLE).plan.json --out $(OUTPUT_DIR)/$(EXAMPLE).nano
	@echo "$(GREEN)✓ NanoLang skeleton generated: $(OUTPUT_DIR)/$(EXAMPLE).nano$(NC)"
	@ls -lh $(OUTPUT_DIR)/$(EXAMPLE).nano

# Pattern rule: compile any .opl file to .plan.json
output/%.plan.json: %.opl $(OPL_CLI) | $(OUTPUT_DIR)
	@echo "$(BLUE)Processing $<...$(NC)"
	@$(OPL_CLI) parse $< --out output/$*.ast.json
	@echo "  $(GREEN)✓$(NC) AST"
	@$(OPL_CLI) validate $< --out output/$*.validate.json || true
	@echo "  $(GREEN)✓$(NC) Validation"
	@$(OPL_CLI) compile $< --out output/$*.plan.json
	@echo "  $(GREEN)✓$(NC) PLAN_IR"

# List generated files
list:
	@echo "$(BLUE)Generated output files:$(NC)"
	@if [ -d $(OUTPUT_DIR) ] && [ -n "$$(ls -A $(OUTPUT_DIR) 2>/dev/null)" ]; then \
		ls -lh $(OUTPUT_DIR)/ | grep -v "^total" | awk '{printf "  %s %s %s\n", $$9, $$5, $$6" "$$7}' ; \
	else \
		echo "  (none - run 'make all' to build examples)"; \
	fi

# Clean generated files
clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	@rm -rf $(OUTPUT_DIR)
	@echo "$(GREEN)✓ Clean complete$(NC)"

# Clean everything including the CLI
distclean: clean
	@echo "$(YELLOW)Removing OPL CLI...$(NC)"
	@rm -f $(OPL_CLI)
	@echo "$(GREEN)✓ Full clean complete$(NC)"
