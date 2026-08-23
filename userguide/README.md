# NanoLang User Guide

This directory contains the source pages selected by `nav.txt`. Generated
reference pages are built from current repository data.

## Local workflow

- Build and validate HTML: `make -C userguide html`
- Run snippet checks: `make userguide-check` (or `make -C userguide check`)
- Serve locally: `make -C userguide serve` (then browse http://localhost:3000)
- Local server behavior matches GitHub Pages and serves `index.html` for `/`.

### Build options

- `USERGUIDE_TIMEOUT=600` sets the HTML build timeout (seconds).
- Generated output is written to `build/userguide/html` and is not committed.
