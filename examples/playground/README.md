# NanoLang Interactive Playground

I run the interpreter in WebAssembly in the browser. A local static server is
still required because browsers do not load the Wasm bundle reliably from a
`file://` URL, and the editor imports CodeMirror from `esm.sh`.

## Features

- **WASM interpreter** — `nanolang.wasm` (Emscripten build of the full nano interpreter)
- **CodeMirror 6 editor** — syntax highlighting (keywords, types, strings, numbers, comments) via a custom StreamLanguage mode; one-dark theme
- **Share/permalink** — 🔗 button encodes source in URL hash (`#v1/<base64url-gzip>`); paste the URL to share a snippet; opened links restore automatically
- **9 built-in examples** — hello world through recursion and structs
- **Download** — save current source as `<example>.nano`

## Run locally

```sh
# Any static file server works:
python3 -m http.server 8000 --directory examples/playground/public
# then open http://localhost:8000
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
