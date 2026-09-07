# NanoLang User Guide

This directory contains the source pages selected by `nav.txt`. Generated
reference pages are built from current repository data.

## Local workflow

- Build and validate HTML (six language editions): `make -C userguide html`
- Run snippet checks: `make userguide-check` (or `make -C userguide check`)
- Serve locally: `make -C userguide serve` (then browse http://localhost:3000)
- Local server behavior matches GitHub Pages and serves `index.html` for `/`.
- Translation drafts: `userguide/i18n/` (`README.md` there is the contributor workflow).

### Build options

- `USERGUIDE_TIMEOUT=600` sets the HTML build timeout (seconds).
- Generated output is written to `build/userguide/html` and is not committed.

### Example evidence

I keep declared output in each runnable example header. `make -C examples test`
compiles the language and verified catalogs, runs finite examples, and checks
their declared output. A platform-gated example that prints `SKIP:` is not a
portable success claim; it records the unavailable boundary.
