# Nanolang Playground

A browser-based interactive playground for Nanolang. Edit code in the browser, click **Run** (or press `Ctrl+Enter`), and see output instantly.

## Quick start

```bash
# From the repo root — build the interpreter first if needed
make build

# Start the playground server (port 8792)
make playground
```

Then open **http://localhost:8792** in your browser.

## Manual start

```bash
node playground/server.mjs
```

The server must be started from the repo root so it can locate `bin/nano`.

## Architecture

```
browser (index.html)
  │  POST /eval  { source, timeout }
  ▼
server.mjs  (Node.js, port 8792)
  │  writes source to temp .nano file
  │  runs  bin/nano <tmpfile>
  │  captures stdout + stderr
  ▼
{ output, error }  (JSON)
```

- **`index.html`** — single-file web app; CodeMirror 6 editor loaded from esm.sh CDN; nanolang syntax highlighting via a custom StreamLanguage tokeniser; Share button encodes source as URL hash (base64).
- **`server.mjs`** — ~60 lines of Node.js; no dependencies beyond the Node stdlib; CORS-enabled for local and AgentFS hosting.

## Share links

The **Share** button encodes the current source as a URL fragment:

```
http://localhost:8792/#<base64-encoded-source>
```

Paste the URL into another browser to restore the code automatically.

## Embedding in AgentFS

The server sets `Access-Control-Allow-Origin: *`, so you can proxy requests or serve the HTML from a different origin. Set `PORT` via environment variable if needed:

```bash
PORT=9000 node playground/server.mjs
```

## Example programs

| Name | Description |
|------|-------------|
| Hello World | Basic `println`, `main` function |
| Fibonacci | Recursive algorithm, `while` loop |
| Match Expression | `union` types, pattern matching |
| Structs | `struct` definition, field access, functions |
| String Interpolation | f-string `f"..."` syntax |
