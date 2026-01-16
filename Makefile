# ============================================================================
# Nanolang Makefile - Portable Wrapper
# ============================================================================
#
# This Makefile detects whether GNU make or BSD make is being used.
# The actual build logic is in Makefile.gnu (which requires GNU make).
#
# On BSD systems: Running 'make' will show a clear error to use 'gmake'
# On Linux/macOS: Running 'make' works (delegates to Makefile.gnu)
#
# ============================================================================

# Check if GNU make is being used and delegate to Makefile.gnu
# This uses only portable syntax that both BSD make and GNU make understand

.PHONY: all build test test-docs examples clean help

all:
	@if $(MAKE) --version 2>/dev/null | grep -q 'GNU Make' 2>/dev/null; then \
		$(MAKE) -f Makefile.gnu all; \
	else \
		printf '\n'; \
		printf '╔═══════════════════════════════════════════════════════════════╗\n'; \
		printf '║                     ❌ ERROR: Wrong Make Tool                  ║\n'; \
		printf '╚═══════════════════════════════════════════════════════════════╝\n'; \
		printf '\n'; \
		printf 'This project requires GNU make, but you are using BSD make.\n'; \
		printf '\n'; \
		printf '📋 SOLUTION:\n'; \
		printf '  Use "gmake" instead of "make" for all build commands.\n'; \
		printf '\n'; \
		printf '🔧 INSTALLATION:\n'; \
		printf '  FreeBSD:  pkg install gmake\n'; \
		printf '  OpenBSD:  pkg_add gmake\n'; \
		printf '  NetBSD:   pkgin install gmake\n'; \
		printf '\n'; \
		printf '📝 EXAMPLES:\n'; \
		printf '  gmake              # Build the project\n'; \
		printf '  gmake test         # Run tests\n'; \
		printf '  gmake clean        # Clean build artifacts\n'; \
		printf '  gmake examples     # Build examples\n'; \
		printf '\n'; \
		exit 1; \
	fi

build test test-docs examples clean help install uninstall:
	@if $(MAKE) --version 2>/dev/null | grep -q 'GNU Make' 2>/dev/null; then \
		$(MAKE) -f Makefile.gnu $@; \
	else \
		printf '\n❌ ERROR: This requires GNU make. Use "gmake" instead of "make".\n\n'; \
		exit 1; \
	fi

# Catch-all for any other targets
%:
	@if $(MAKE) --version 2>/dev/null | grep -q 'GNU Make' 2>/dev/null; then \
		$(MAKE) -f Makefile.gnu $@; \
	else \
		printf '\n❌ ERROR: This requires GNU make. Use "gmake" instead of "make".\n\n'; \
		exit 1; \
	fi
