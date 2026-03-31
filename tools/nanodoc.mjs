#!/usr/bin/env node
/**
 * nanodoc — mdoc-style HTML API documentation generator for .nano source files
 *
 * Usage:
 *   node tools/nanodoc.mjs [--output <dir>] <file.nano> ...
 *   node tools/nanodoc.mjs --output docs/api std/http/http.nano std/datetime/datetime.nano
 *
 * Doc comment syntax (triple-slash, must immediately precede a declaration):
 *   /// Brief one-line description.
 *   ///
 *   /// Longer paragraph text.
 *   ///
 *   /// @param name   Description of parameter.
 *   /// @returns      Description of return value.
 *   /// @example
 *   ///   let r: HttpResponse = (http::get "https://example.com")
 *
 * Documented declarations:
 *   pub fn name(params) -> RetType        — public function
 *   pub struct/enum/union Name { ... }   — public named type
 *   struct/enum/union Name { ... }       — module-private named type
 *   opaque type Name                     — opaque FFI type
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);
let outputDir = 'docs/api';
const inputFiles = [];

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--output' || args[i] === '-o') {
    outputDir = args[++i];
  } else if (args[i].startsWith('--output=')) {
    outputDir = args[i].slice('--output='.length);
  } else {
    inputFiles.push(args[i]);
  }
}

if (inputFiles.length === 0) {
  console.error('nanodoc: no input files specified');
  console.error('Usage: node tools/nanodoc.mjs [--output <dir>] <file.nano> ...');
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} DocParam
 * @property {string} name
 * @property {string} description
 */

/**
 * @typedef {Object} DocComment
 * @property {string}     brief      - First non-empty line
 * @property {string[]}   body       - Subsequent description lines
 * @property {DocParam[]} params
 * @property {string}     returns
 * @property {string[]}   example    - Lines of code example
 */

/**
 * @typedef {Object} Item
 * @property {'fn'|'struct'|'enum'|'union'|'opaque'} kind
 * @property {string}     name
 * @property {boolean}    pub
 * @property {string}     signature  - Full declaration line (sanitized)
 * @property {DocComment|null} doc
 */

/**
 * @typedef {Object} Module
 * @property {string}   name        - Module name (filename stem)
 * @property {string}   filePath    - Resolved input path
 * @property {string}   description - Module-level description from leading # comments
 * @property {Item[]}   items
 */

/**
 * Parse a `///` doc-comment block into structured fields.
 * @param {string[]} lines
 * @returns {DocComment}
 */
function parseDocComment(lines) {
  const stripped = lines.map(l => {
    const m = l.match(/^\/\/\/(.*)$/);
    return m ? m[1].replace(/^ /, '') : '';
  });

  const doc = { brief: '', body: [], params: [], returns: '', example: [] };
  let inExample = false;
  let briefDone = false;

  for (const line of stripped) {
    if (!briefDone) {
      if (line.trim() === '') continue;  // skip leading blank lines
      doc.brief = line.trim();
      briefDone = true;
      continue;
    }

    if (inExample) {
      // Collect example until next @tag or end
      if (line.trim().startsWith('@')) {
        inExample = false;
        // fall through to handle the @tag below
      } else {
        doc.example.push(line);
        continue;
      }
    }

    const paramMatch = line.match(/^@param\s+(\S+)\s*(.*)/);
    const returnsMatch = line.match(/^@returns?\s*(.*)/);
    const exampleMatch = line.trim() === '@example';

    if (paramMatch) {
      doc.params.push({ name: paramMatch[1], description: paramMatch[2].trim() });
    } else if (returnsMatch) {
      doc.returns = returnsMatch[1].trim();
    } else if (exampleMatch) {
      inExample = true;
    } else {
      doc.body.push(line);
    }
  }

  // Trim trailing blank lines from body
  while (doc.body.length > 0 && doc.body[doc.body.length - 1].trim() === '') {
    doc.body.pop();
  }
  // Trim trailing blank lines from example
  while (doc.example.length > 0 && doc.example[doc.example.length - 1].trim() === '') {
    doc.example.pop();
  }

  return doc;
}

/**
 * Parse a .nano source file and extract documented items.
 * @param {string} filePath
 * @returns {Module}
 */
function parseNanoFile(filePath) {
  const src = readFileSync(filePath, 'utf8');
  const lines = src.split('\n');

  const name = basename(filePath, '.nano');
  const items = [];

  // Extract module description from leading # comment block at top of file
  const descLines = [];
  let i = 0;
  while (i < lines.length) {
    const l = lines[i].trim();
    if (l === '') { i++; continue; }
    if (l.startsWith('#')) {
      descLines.push(l.slice(1).trim());
      i++;
    } else {
      break;
    }
  }
  // Skip a leading title line like "# HTTP Client Library for Nanolang" — keep rest as description
  const description = descLines.join(' ').replace(/\s+/g, ' ').trim();

  // Main scan
  let pendingDoc = [];   // accumulated /// lines
  let lineIdx = 0;

  while (lineIdx < lines.length) {
    const raw = lines[lineIdx];
    const trimmed = raw.trim();

    // Accumulate /// doc-comment lines
    if (/^\/\/\//.test(trimmed)) {
      pendingDoc.push(trimmed);
      lineIdx++;
      continue;
    }

    // Blank line resets pending doc (doc must directly precede declaration)
    if (trimmed === '') {
      pendingDoc = [];
      lineIdx++;
      continue;
    }

    // Skip shadow blocks (test-only, not API)
    if (/^shadow\s+\w+/.test(trimmed)) {
      pendingDoc = [];
      // skip to matching closing brace
      let depth = 0;
      while (lineIdx < lines.length) {
        const t = lines[lineIdx].trim();
        for (const ch of t) {
          if (ch === '{') depth++;
          if (ch === '}') depth--;
        }
        lineIdx++;
        if (depth <= 0) break;
      }
      continue;
    }

    // Skip extern fn (low-level FFI, not part of public API docs)
    if (/^extern\s+fn\s+/.test(trimmed)) {
      pendingDoc = [];
      lineIdx++;
      continue;
    }

    // Match pub fn
    let m = trimmed.match(/^pub\s+fn\s+(\w+)\(([^)]*)\)\s*->\s*(\S+)/);
    if (m) {
      const [, fnName, rawParams, retType] = m;
      // Normalize parameter list: strip leading underscores (FFI convention)
      const params = rawParams
        .split(',')
        .map(p => p.trim())
        .filter(Boolean)
        .map(p => p.replace(/^_/, ''));  // strip leading _ from param names
      const signature = `pub fn ${fnName}(${params.join(', ')}) -> ${retType}`;
      items.push({
        kind: 'fn',
        name: fnName,
        pub: true,
        signature,
        doc: pendingDoc.length > 0 ? parseDocComment(pendingDoc) : null,
      });
      pendingDoc = [];
      lineIdx++;
      continue;
    }

    // Match private fn (not pub) — skip, not documented
    m = trimmed.match(/^fn\s+(\w+)\(/);
    if (m) {
      pendingDoc = [];
      lineIdx++;
      continue;
    }

    // Match opaque type
    m = trimmed.match(/^opaque\s+type\s+(\w+)/);
    if (m) {
      items.push({
        kind: 'opaque',
        name: m[1],
        pub: true,
        signature: `opaque type ${m[1]}`,
        doc: pendingDoc.length > 0 ? parseDocComment(pendingDoc) : null,
      });
      pendingDoc = [];
      lineIdx++;
      continue;
    }

    // Match pub struct/enum/union
    m = trimmed.match(/^pub\s+(struct|enum|union)\s+(\w+)/);
    if (m) {
      items.push({
        kind: m[1],
        name: m[2],
        pub: true,
        signature: `pub ${m[1]} ${m[2]} { ... }`,
        doc: pendingDoc.length > 0 ? parseDocComment(pendingDoc) : null,
      });
      pendingDoc = [];
      lineIdx++;
      continue;
    }

    // Match module-private struct/enum/union
    m = trimmed.match(/^(struct|enum|union)\s+(\w+)/);
    if (m) {
      items.push({
        kind: m[1],
        name: m[2],
        pub: false,
        signature: `${m[1]} ${m[2]} { ... }`,
        doc: pendingDoc.length > 0 ? parseDocComment(pendingDoc) : null,
      });
      pendingDoc = [];
      lineIdx++;
      continue;
    }

    // Any other non-blank, non-comment line clears pending doc
    if (!trimmed.startsWith('#') && !trimmed.startsWith('/*')) {
      pendingDoc = [];
    }

    lineIdx++;
  }

  return { name, filePath, description, items };
}

// ---------------------------------------------------------------------------
// HTML generator
// ---------------------------------------------------------------------------

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

const CSS = `
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #f8f8f6;
  --bg-sidebar: #1e1e2e;
  --text: #1a1a2e;
  --text-dim: #555;
  --text-sidebar: #cdd6f4;
  --accent: #7c5cbf;
  --accent-light: #ede9f8;
  --code-bg: #1e1e2e;
  --code-text: #cdd6f4;
  --border: #ddd;
  --sidebar-width: 240px;
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', ui-monospace, monospace;
  --font-sans: system-ui, -apple-system, sans-serif;
}

body {
  font-family: var(--font-sans);
  background: var(--bg);
  color: var(--text);
  display: flex;
  min-height: 100vh;
  line-height: 1.6;
}

/* Sidebar */
.sidebar {
  width: var(--sidebar-width);
  min-width: var(--sidebar-width);
  background: var(--bg-sidebar);
  color: var(--text-sidebar);
  padding: 1.5rem 1rem;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}

.sidebar h1 {
  font-size: 1rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 1.25rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.sidebar h2 {
  font-size: 0.7rem;
  font-weight: 600;
  color: #89b4fa;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 1.25rem 0 0.4rem;
}

.sidebar a {
  display: block;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--text-sidebar);
  text-decoration: none;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.sidebar a:hover { background: rgba(255,255,255,0.1); color: #fff; }
.sidebar a.active { background: var(--accent); color: #fff; }
.sidebar a.fn-link::before { content: 'fn '; color: #89b4fa; }
.sidebar a.type-link::before { content: 'type '; color: #a6e3a1; }

/* Main content */
main {
  flex: 1;
  padding: 2rem 3rem;
  max-width: 900px;
}

.module-header {
  border-bottom: 2px solid var(--border);
  padding-bottom: 1rem;
  margin-bottom: 2rem;
}

.module-header h1 {
  font-family: var(--font-mono);
  font-size: 1.8rem;
  color: var(--accent);
}

.module-header p { margin-top: 0.5rem; color: var(--text-dim); }

/* Section headings */
.items-section h2 {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4rem;
  margin: 2rem 0 1rem;
}

/* Individual item */
.item {
  margin-bottom: 2rem;
  scroll-margin-top: 1rem;
}

.item-sig {
  background: var(--code-bg);
  color: var(--code-text);
  font-family: var(--font-mono);
  font-size: 0.9rem;
  padding: 0.6rem 1rem;
  border-radius: 6px;
  border-left: 3px solid var(--accent);
  overflow-x: auto;
  white-space: pre;
}

.item-sig .kw   { color: #cba6f7; }
.item-sig .name { color: #89dceb; font-weight: 600; }
.item-sig .type { color: #a6e3a1; }
.item-sig .punct { color: #6c7086; }

.item-doc {
  padding: 0.6rem 0 0 0.2rem;
}

.item-brief {
  font-size: 0.95rem;
  margin-bottom: 0.5rem;
}

.item-body p {
  font-size: 0.9rem;
  color: var(--text-dim);
  margin-bottom: 0.4rem;
}

.item-params {
  margin: 0.75rem 0;
  font-size: 0.88rem;
}

.item-params table { border-collapse: collapse; }
.item-params th {
  text-align: left;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-dim);
  padding: 0 0.75rem 0.3rem 0;
}
.item-params td { padding: 0.15rem 0.75rem 0.15rem 0; vertical-align: top; }
.item-params .pname { font-family: var(--font-mono); font-size: 0.82rem; color: var(--accent); }
.item-params .pdesc { color: var(--text-dim); }
.item-returns {
  font-size: 0.88rem;
  margin: 0.5rem 0;
  color: var(--text-dim);
}
.item-returns .ret-label { font-weight: 600; color: var(--text); }

.item-example {
  margin-top: 0.75rem;
}
.item-example .ex-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-dim);
  margin-bottom: 0.3rem;
}
.item-example pre {
  background: var(--code-bg);
  color: var(--code-text);
  font-family: var(--font-mono);
  font-size: 0.83rem;
  padding: 0.75rem 1rem;
  border-radius: 6px;
  overflow-x: auto;
}

/* Index page */
.module-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1rem;
  margin-top: 1.5rem;
}

.module-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.25rem;
  background: #fff;
  text-decoration: none;
  color: inherit;
  transition: box-shadow 0.15s;
}

.module-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.1); }
.module-card h3 {
  font-family: var(--font-mono);
  font-size: 1rem;
  color: var(--accent);
  margin-bottom: 0.3rem;
}
.module-card p { font-size: 0.85rem; color: var(--text-dim); }
.module-card .item-count {
  font-size: 0.75rem;
  color: var(--text-dim);
  margin-top: 0.5rem;
}

/* Breadcrumb */
.breadcrumb {
  font-size: 0.8rem;
  color: var(--text-dim);
  margin-bottom: 1.5rem;
}
.breadcrumb a { color: var(--accent); text-decoration: none; }
.breadcrumb a:hover { text-decoration: underline; }

/* Responsive */
@media (max-width: 700px) {
  body { flex-direction: column; }
  .sidebar { width: 100%; height: auto; position: static; }
  main { padding: 1.25rem; }
}
`.trim();

/**
 * Highlight a nano declaration signature using span tags.
 * Very simple tokenizer just for pub fn / struct / opaque type signatures.
 */
function highlightSig(sig) {
  const keywords = ['pub', 'fn', 'opaque', 'type', 'struct', 'enum', 'union', 'extern'];
  // Tokenize by whitespace/punctuation
  return esc(sig)
    .replace(/\b(pub|fn|opaque|type|struct|enum|union)\b/g, '<span class="kw">$1</span>')
    .replace(/(?<=(?:fn|struct|enum|union)\s)<span[^>]*>([^<]+)<\/span>|(?<=(?:fn|struct|enum|union)\s)(\w+)/g, (match) => {
      // This is too complex inline — just do simple keyword coloring
      return match;
    });
}

/**
 * Simpler approach: highlight signature by splitting on spaces and parentheses.
 */
function renderSig(sig) {
  const keywords = new Set(['pub', 'fn', 'opaque', 'type', 'struct', 'enum', 'union']);
  // Split into tokens preserving delimiters
  const tokens = sig.split(/(\s+|[(),->]+)/);
  let firstWord = true;
  let afterKw = false;
  return tokens.map(tok => {
    if (/^\s+$/.test(tok) || !tok) return esc(tok);
    if (keywords.has(tok)) {
      afterKw = true;
      return `<span class="kw">${esc(tok)}</span>`;
    }
    if (afterKw && /^\w+$/.test(tok)) {
      afterKw = false;
      return `<span class="name">${esc(tok)}</span>`;
    }
    // Type annotations after `:`  and `->` get type color
    if (/^[A-Z][a-zA-Z0-9_]*$/.test(tok) || tok === 'int' || tok === 'string' || tok === 'bool' || tok === 'float' || tok === 'void') {
      return `<span class="type">${esc(tok)}</span>`;
    }
    if (/^[(),->]+$/.test(tok)) {
      return `<span class="punct">${esc(tok)}</span>`;
    }
    afterKw = false;
    return esc(tok);
  }).join('');
}

/**
 * Render a DocComment to HTML snippet (no wrapper div).
 */
function renderDoc(doc) {
  if (!doc) return '';
  let html = '';

  if (doc.brief) {
    html += `<p class="item-brief">${esc(doc.brief)}</p>`;
  }

  if (doc.body.length > 0) {
    html += '<div class="item-body">';
    for (const line of doc.body) {
      if (line.trim() === '') {
        html += '<p></p>';
      } else {
        html += `<p>${esc(line)}</p>`;
      }
    }
    html += '</div>';
  }

  if (doc.params.length > 0) {
    html += '<div class="item-params"><table><tr><th>Parameter</th><th>Description</th></tr>';
    for (const p of doc.params) {
      html += `<tr><td class="pname">${esc(p.name)}</td><td class="pdesc">${esc(p.description)}</td></tr>`;
    }
    html += '</table></div>';
  }

  if (doc.returns) {
    html += `<p class="item-returns"><span class="ret-label">Returns:</span> ${esc(doc.returns)}</p>`;
  }

  if (doc.example.length > 0) {
    html += '<div class="item-example"><div class="ex-label">Example</div>';
    html += `<pre>${esc(doc.example.join('\n'))}</pre>`;
    html += '</div>';
  }

  return html;
}

/**
 * Render a complete module page as HTML string.
 * @param {Module} mod
 * @param {Module[]} allModules - for sidebar
 * @returns {string}
 */
function renderModulePage(mod, allModules) {
  const types = mod.items.filter(it => it.kind !== 'fn');
  const fns   = mod.items.filter(it => it.kind === 'fn');

  const sidebarModules = allModules.map(m => {
    const cls = m.name === mod.name ? ' class="active"' : '';
    return `<a href="${esc(m.name)}.html"${cls}>${esc(m.name)}</a>`;
  }).join('\n');

  const sidebarFns = fns.map(fn =>
    `<a href="#fn-${esc(fn.name)}" class="fn-link">${esc(fn.name)}</a>`
  ).join('\n');

  const sidebarTypes = types.map(t =>
    `<a href="#type-${esc(t.name)}" class="type-link">${esc(t.name)}</a>`
  ).join('\n');

  const typesHtml = types.length === 0 ? '' : `
<section class="items-section">
  <h2>Types</h2>
  ${types.map(item => {
    const anchor = `type-${item.name}`;
    const badge = item.pub ? '' : ' <small style="color:#aaa">(private)</small>';
    return `
  <div class="item" id="${esc(anchor)}">
    <div class="item-sig">${renderSig(item.signature)}</div>
    <div class="item-doc">
      ${renderDoc(item.doc)}${!item.doc ? `<p style="color:#aaa;font-size:0.85rem;font-style:italic">No documentation.</p>` : ''}
    </div>
  </div>`;
  }).join('')}
</section>`;

  const fnsHtml = fns.length === 0 ? '' : `
<section class="items-section">
  <h2>Functions</h2>
  ${fns.map(item => {
    const anchor = `fn-${item.name}`;
    return `
  <div class="item" id="${esc(anchor)}">
    <div class="item-sig">${renderSig(item.signature)}</div>
    <div class="item-doc">
      ${renderDoc(item.doc)}${!item.doc ? `<p style="color:#aaa;font-size:0.85rem;font-style:italic">No documentation.</p>` : ''}
    </div>
  </div>`;
  }).join('')}
</section>`;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${esc(mod.name)} — nanolang API</title>
  <style>${CSS}</style>
</head>
<body>
<nav class="sidebar">
  <h1>nanolang</h1>
  <h2>Modules</h2>
  ${sidebarModules}
  ${sidebarTypes.length ? `<h2>Types</h2>\n  ${sidebarTypes}` : ''}
  ${sidebarFns.length ? `<h2>Functions</h2>\n  ${sidebarFns}` : ''}
</nav>
<main>
  <div class="breadcrumb"><a href="index.html">API Reference</a> / ${esc(mod.name)}</div>
  <div class="module-header">
    <h1>${esc(mod.name)}</h1>
    ${mod.description ? `<p>${esc(mod.description)}</p>` : ''}
  </div>
  ${typesHtml}
  ${fnsHtml}
</main>
</body>
</html>`;
}

/**
 * Render the index page listing all modules.
 * @param {Module[]} modules
 * @returns {string}
 */
function renderIndexPage(modules) {
  const sidebarModules = modules.map(m =>
    `<a href="${esc(m.name)}.html">${esc(m.name)}</a>`
  ).join('\n');

  const cards = modules.map(mod => {
    const fnCount = mod.items.filter(it => it.kind === 'fn').length;
    const typeCount = mod.items.filter(it => it.kind !== 'fn').length;
    const counts = [
      fnCount ? `${fnCount} fn` : '',
      typeCount ? `${typeCount} type${typeCount !== 1 ? 's' : ''}` : '',
    ].filter(Boolean).join(', ');
    return `
  <a class="module-card" href="${esc(mod.name)}.html">
    <h3>${esc(mod.name)}</h3>
    ${mod.description ? `<p>${esc(mod.description.slice(0, 120))}${mod.description.length > 120 ? '…' : ''}</p>` : '<p style="color:#aaa;font-style:italic">No description.</p>'}
    <div class="item-count">${counts}</div>
  </a>`;
  }).join('');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>nanolang API Reference</title>
  <style>${CSS}</style>
</head>
<body>
<nav class="sidebar">
  <h1>nanolang</h1>
  <h2>Modules</h2>
  ${sidebarModules}
</nav>
<main>
  <div class="module-header">
    <h1>API Reference</h1>
    <p>Documentation for the nanolang standard library and modules.</p>
  </div>
  <div class="module-grid">
    ${cards}
  </div>
</main>
</body>
</html>`;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

mkdirSync(outputDir, { recursive: true });

const modules = inputFiles.map(f => {
  try {
    const mod = parseNanoFile(f);
    console.log(`nanodoc: ${f} → ${mod.name} (${mod.items.length} items)`);
    return mod;
  } catch (err) {
    console.error(`nanodoc: error reading ${f}: ${err.message}`);
    process.exit(1);
  }
});

// Write per-module pages
for (const mod of modules) {
  const html = renderModulePage(mod, modules);
  const outPath = join(outputDir, `${mod.name}.html`);
  writeFileSync(outPath, html, 'utf8');
  console.log(`nanodoc: wrote ${outPath}`);
}

// Write index page
const indexHtml = renderIndexPage(modules);
const indexPath = join(outputDir, 'index.html');
writeFileSync(indexPath, indexHtml, 'utf8');
console.log(`nanodoc: wrote ${indexPath}`);

console.log(`nanodoc: done — ${modules.length} module(s) documented in ${outputDir}/`);
