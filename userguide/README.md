# NanoLang User Guide

This directory is the progressive NanoLang user guide.

## Local workflow

- Build HTML: `make -C userguide html`
- Run snippet checks: `make userguide-check` (or `make -C userguide check`)
- Serve locally: `make -C userguide serve` (then browse http://localhost:3000)
- Local server behavior matches GitHub Pages and serves `index.html` for `/`.
