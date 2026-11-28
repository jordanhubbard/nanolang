#!/usr/bin/env python3
"""I preserve the original private GNU2.42 diagnostic entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ci_build_capture_assembler import main

if __name__ == '__main__':
    main()
