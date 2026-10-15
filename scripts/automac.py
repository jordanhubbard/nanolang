#!/usr/bin/env python3
"""I retain the old command name as a GitHub Issues compatibility entry point."""
import sys
from autogithub import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
