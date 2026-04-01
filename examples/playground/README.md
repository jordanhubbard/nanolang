# NanoLang Interactive Playground

A zero-install, browser-based NanoLang editor that runs the interpreter
entirely via WebAssembly — no server needed for execution.

## Features

- **WASM interpreter** — `nanolang.wasm` (Emscripten build of the full nano interpreter)
- **CodeMirror 6 editor** — syntax highlighting (keywords, types, strings, numbers, comments) via a custom StreamLanguage mode; one-dark theme
- **Share/permalink** — 🔗 button encodes source in URL hash (`#v1/<base64url-gzip>`); paste the URL to share a snippet; opened links restore automatically
- **Format on run** — integrated with the nano-fmt output path
- **9 built-in examples** — hello world through recursion and structs
- **Download** — save current source as `<example>.nano`

## Run locally

```sh
# Any static file server works:
python3 -m http.server 8000 --directory examples/playground/public
# then open http://localhost:8000
```

## Host on AgentFS (sparky:8791)

AgentFS serves static files from its content-addressed store. To publish:

```sh
# Upload the playground directory
curl -X PUT http://100.87.229.125:8791/agentos/playground/ \
  -H "Content-Type: application/x-directory" \
  --data-binary @- < /dev/null

# Upload each file (repeat for index.html, app.js, examples.js, style.css, nanolang.js, nanolang.wasm)
for f in examples/playground/public/*; do
  curl -X PUT "http://100.87.229.125:8791/agentos/playground/$(basename $f)" \
    --data-binary "@$f"
done

# Access at:
# http://100.87.229.125:8791/agentos/playground/index.html
```

## Build (update nanolang.wasm)

The WASM binary is built from the nanolang interpreter via Emscripten:

```sh
# Requires emscripten SDK (source emsdk/emsdk_env.sh first)
make -f Makefile.gnu wasm-playground
# Outputs: examples/playground/public/nanolang.wasm + nanolang.js
```

## Architecture

```
index.html   — shell + CodeMirror 6 ES module (CDN, no build step)
app.js       — WASM init + run/check loop + share/permalink + CM glue
examples.js  — built-in example snippets
nanolang.js  — Emscripten JS glue (auto-generated)
nanolang.wasm— Emscripten WASM binary (~360KB)
style.css    — layout + dark theme
```

## Permalink format

`#v1/<base64url(gzip(source))>` — compresses with `CompressionStream('gzip')`,
encodes as URL-safe base64 (no padding).  Falls back to `#v0/<base64url(source)>`
if CompressionStream is unavailable (old browsers).
