# ProTracker Clone - Multi-file Project Makefile
# Demonstrates compiling a nanolang project from multiple source files

# Compiler settings
NANOC = ../../bin/nanoc
NANO_MODULE_PATH = ../../modules
export NANO_MODULE_PATH

# Project settings
PROJECT_NAME = protracker
MAIN_FILE = main.nano
OUTPUT = ../../bin/$(PROJECT_NAME)

# Source files (all .nano files in project)
SOURCES = types.nano \
          pattern.nano \
          ui.nano \
          main.nano

# Build target
all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	@echo "════════════════════════════════════════════════════════"
	@echo " Building ProTracker Clone (multi-file project)"
	@echo "════════════════════════════════════════════════════════"
	@echo ""
	@echo "Source files:"
	@for file in $(SOURCES); do \
		echo "  - $$file ($(shell wc -l < $$file) lines)"; \
	done
	@echo ""
	@echo "Compiling $(MAIN_FILE) (imports all modules)..."
	cd ../.. && NANO_MODULE_PATH=modules ./bin/nanoc examples/protracker/$(MAIN_FILE) -o $(OUTPUT)
	@echo ""
	@echo "✓ ProTracker built: $(OUTPUT)"
	@echo ""
	@echo "Run with:"
	@echo "  $(OUTPUT)"
	@echo ""

# Run the compiled binary
run: $(OUTPUT)
	@echo "Running ProTracker..."
	$(OUTPUT)

# Clean build artifacts
clean:
	@echo "Cleaning ProTracker build artifacts..."
	rm -f $(OUTPUT)
	@echo "✓ Clean complete"

# Development: watch for changes and rebuild
watch:
	@echo "Watching for changes (Ctrl+C to stop)..."
	@while true; do \
		make -q all || make all; \
		sleep 2; \
	done

# Show project statistics
stats:
	@echo "ProTracker Project Statistics"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Source files:"
	@for file in $(SOURCES); do \
		lines=$$(wc -l < $$file); \
		echo "  $$file: $$lines lines"; \
	done
	@echo ""
	@total=0; \
	for file in $(SOURCES); do \
		lines=$$(wc -l < $$file); \
		total=$$((total + lines)); \
	done; \
	echo "Total lines: $$total"
	@echo ""
	@echo "Modules imported:"
	@grep -h "^import" $(SOURCES) | sort -u
	@echo ""

# Help target
help:
	@echo "ProTracker Clone - Makefile Commands"
	@echo "═══════════════════════════════════════"
	@echo ""
	@echo "  make          - Build ProTracker"
	@echo "  make run      - Build and run"
	@echo "  make clean    - Remove built files"
	@echo "  make watch    - Auto-rebuild on changes"
	@echo "  make stats    - Show project statistics"
	@echo "  make help     - Show this help"
	@echo ""

.PHONY: all run clean watch stats help
