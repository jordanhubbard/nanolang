# ============================================================================
# Nanolang Makefile - BSD/gmake bridge
# ============================================================================
#
# GNU make will prefer GNUmakefile automatically.
# BSD make reads this file and reinvokes gmake on GNUmakefile.
#
# ============================================================================

.PHONY: all build test test-docs examples clean help coverage coverage-report module-package-audit

all:
	@if $(MAKE) --version 2>/dev/null | grep -q 'GNU Make' 2>/dev/null; then \
		$(MAKE) -f GNUmakefile all; \
	elif command -v gmake >/dev/null 2>&1; then \
		bridge_dir="$$(pwd)"; \
		if [ ! -f "$$bridge_dir/GNUmakefile" ] && [ -f "$$bridge_dir/../GNUmakefile" ]; then \
			bridge_dir="$$bridge_dir/.."; \
		fi; \
		exec gmake -C "$$bridge_dir" -f GNUmakefile all; \
	else \
		printf '\nERROR: GNU make is required but "gmake" was not found.\n'; \
		printf 'Install GNU make and retry:\n'; \
		printf '  FreeBSD: pkg install gmake\n'; \
		printf '  OpenBSD: pkg_add gmake\n'; \
		printf '  NetBSD:  pkgin install gmake\n\n'; \
		exit 1; \
	fi

build test test-docs test-doc-md examples clean help install uninstall coverage coverage-report module-package-audit wasm-playground wasm-playground-clean test-ringbuf:
	@if $(MAKE) --version 2>/dev/null | grep -q 'GNU Make' 2>/dev/null; then \
		$(MAKE) -f GNUmakefile $@; \
	elif command -v gmake >/dev/null 2>&1; then \
		bridge_dir="$$(pwd)"; \
		if [ ! -f "$$bridge_dir/GNUmakefile" ] && [ -f "$$bridge_dir/../GNUmakefile" ]; then \
			bridge_dir="$$bridge_dir/.."; \
		fi; \
		exec gmake -C "$$bridge_dir" -f GNUmakefile $@; \
	else \
		printf '\nERROR: GNU make is required but "gmake" was not found.\n\n'; \
		exit 1; \
	fi

# Catch-all for any other targets
.DEFAULT:
	@if $(MAKE) --version 2>/dev/null | grep -q 'GNU Make' 2>/dev/null; then \
		$(MAKE) -f GNUmakefile $@; \
	elif command -v gmake >/dev/null 2>&1; then \
		bridge_dir="$$(pwd)"; \
		if [ ! -f "$$bridge_dir/GNUmakefile" ] && [ -f "$$bridge_dir/../GNUmakefile" ]; then \
			bridge_dir="$$bridge_dir/.."; \
		fi; \
		exec gmake -C "$$bridge_dir" -f GNUmakefile $@; \
	else \
		printf '\nERROR: GNU make is required but "gmake" was not found.\n\n'; \
		exit 1; \
	fi
