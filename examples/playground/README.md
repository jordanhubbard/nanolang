# NanoLang Interactive Playground

I retain a historical interpreter bundle in WebAssembly for this browser UI.
I do not claim that this bundle implements my current compiler or language.
A local static server is
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

I do not provide a working rebuild of this historical interpreter bundle from
my current source tree. `make wasm-playground` refuses before changing the
bundle; its former source list named removed files and omitted the browser
entry points. Installing Emscripten alone does not repair that build.

My separate `make test-nvm2wasm` gate exercises the supported scalar NanoISA
translator. It does not build this browser interpreter, provide its JavaScript
API, or establish support for every playground example. I retain full backend
and browser integration work as separate roadmap acceptance.

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
