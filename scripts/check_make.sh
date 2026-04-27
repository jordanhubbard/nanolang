#!/bin/sh
# Check if GNU make is being used
# This script is called from the Makefile to provide a helpful error on BSD

if make --version 2>/dev/null | grep -q GNU; then
    exit 0  # GNU make detected, continue
else
    cat << 'EOF'

╔═══════════════════════════════════════════════════════════════╗
║                    ❌ ERROR: Wrong Make Tool                  ║
╚═══════════════════════════════════════════════════════════════╝

This project requires GNU make, but you're using BSD make.

📋 SOLUTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

On BSD systems (FreeBSD/OpenBSD/NetBSD):
  Use 'gmake' instead of 'make'

🔧 INSTALLATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FreeBSD:  pkg install gmake
OpenBSD:  pkg_add gmake  
NetBSD:   pkgin install gmake

📝 AFTER INSTALLING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use 'gmake' for all build commands:
  gmake              # Build the project
  gmake test         # Run tests
  gmake clean        # Clean build artifacts
  gmake examples     # Build examples

💡 WHY GNU MAKE IS REQUIRED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This Makefile uses GNU make features:
  • $(shell ...) for OS detection
  • $(patsubst ...) for path transformations
  • ifeq/ifneq conditionals

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF
    exit 1
fi
