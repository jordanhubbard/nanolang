# NanoLang User Guide

This directory is the progressive NanoLang user guide.

## Local workflow

- Build HTML: `make -C userguide html`
- Run snippet checks: `make userguide-check` (or `make -C userguide check`)
- Serve locally: `make -C userguide serve` (then browse http://localhost:3000)
- Local server behavior matches GitHub Pages and serves `index.html` for `/`.

### Build options

- `USERGUIDE_TIMEOUT=600` sets the HTML build timeout (seconds).
- `USERGUIDE_BUILD_API_DOCS=1` forces API reference regeneration.
- `NANO_USERGUIDE_HIGHLIGHT=0` disables syntax highlighting (enabled by default after 26x speedup).
- `NANO_USERGUIDE_TRACE=1` enables detailed build logging.