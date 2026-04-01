#!/usr/bin/env python3
"""
build_stdlib_docs.py — nanolang stdlib documentation site generator

Runs `nanoc --reflect` on every modules/std/**/*.nano file, collects
the JSON API descriptions, and emits a multi-page static HTML site:

  site/stdlib/index.html           — module index with search bar
  site/stdlib/<module>/index.html  — per-module API page
  site/stdlib/search_index.json    — client-side search index

Optional: if --agentfs-url is given (e.g. http://sparky:8791), the
generated site is PUT to <url>/docs/nanolang/ for network access.

Usage:
  python3 scripts/build_stdlib_docs.py [--nanoc bin/nanoc] [--out site/stdlib]
                                        [--agentfs-url http://sparky:8791]
                                        [--verbose]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build nanolang stdlib docs site")
    p.add_argument("--nanoc",        default="bin/nanoc",       help="path to nanoc binary")
    p.add_argument("--stdlib-dir",   default="modules/std",     help="stdlib root directory")
    p.add_argument("--out",          default="site/stdlib",     help="output directory")
    p.add_argument("--agentfs-url",  default=None,              help="AgentFS base URL for publishing")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()

# ── Reflect a single .nano module ────────────────────────────────────────────

def reflect_module(nanoc: str, nano_path: str, verbose: bool) -> dict | None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = tf.name
    try:
        result = subprocess.run(
            [nanoc, nano_path, "--reflect", tmp],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0 or not os.path.exists(tmp):
            if verbose:
                print(f"  [warn] reflect failed for {nano_path}: {result.stderr[:120]}")
            return None
        with open(tmp) as f:
            data = json.load(f)
        return data
    except Exception as e:
        if verbose:
            print(f"  [warn] {nano_path}: {e}")
        return None
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

# ── Parse doc-comment from .nano source ──────────────────────────────────────

def extract_module_doc(nano_path: str) -> str:
    """Extract leading /* ... */ block comment as module description."""
    try:
        with open(nano_path) as f:
            src = f.read(4096)
    except Exception:
        return ""
    m = re.match(r'\s*/\*+\s*(.*?)\s*\*+/', src, re.DOTALL)
    if m:
        text = m.group(1).strip()
        # Remove leading " * " from each line
        lines = [re.sub(r'^\s*\*\s?', '', l) for l in text.splitlines()]
        return " ".join(l.strip() for l in lines if l.strip())[:300]
    return ""

# ── HTML helpers ─────────────────────────────────────────────────────────────

CSS = """
:root{--bg:#0d1117;--bg2:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;
--accent:#58a6ff;--kw:#ff7b72;--tp:#79c0ff;--fn:#d2a8ff;--badge-bg:#21262d}
*{box-sizing:border-box;margin:0;padding:0}
body{font:15px/1.6 'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text)}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
header{background:var(--bg2);border-bottom:1px solid var(--border);padding:12px 24px;
  display:flex;align-items:center;gap:16px}
header h1{font-size:1.15rem;font-weight:600}
header .ver{color:var(--muted);font-size:.85rem}
.layout{display:flex;min-height:calc(100vh - 48px)}
.sidebar{width:220px;flex-shrink:0;border-right:1px solid var(--border);
  padding:16px 0;background:var(--bg2);position:sticky;top:0;height:calc(100vh - 48px);
  overflow-y:auto}
.sidebar a{display:block;padding:4px 20px;color:var(--text);font-size:.9rem;border-radius:4px}
.sidebar a:hover,.sidebar a.active{background:#1f2937;color:var(--accent)}
.sidebar .group{font-size:.75rem;font-weight:600;color:var(--muted);
  padding:12px 20px 4px;text-transform:uppercase;letter-spacing:.08em}
main{flex:1;padding:32px 48px;max-width:900px}
h2{font-size:1.3rem;margin:0 0 8px;color:var(--text)}
h3{font-size:1rem;margin:24px 0 6px;color:var(--fn)}
.mod-desc{color:var(--muted);margin-bottom:24px;font-size:.9rem}
.fn-card{background:var(--bg2);border:1px solid var(--border);border-radius:8px;
  padding:14px 18px;margin:10px 0}
.fn-sig{font-family:'Fira Code',monospace;font-size:.88rem;white-space:pre-wrap;word-break:break-all}
.fn-sig .kw{color:var(--kw)} .fn-sig .fn{color:var(--fn)} .fn-sig .tp{color:var(--tp)}
.badge{display:inline-block;background:var(--badge-bg);border:1px solid var(--border);
  border-radius:12px;font-size:.72rem;padding:1px 8px;color:var(--muted);margin-left:8px}
.search-wrap{margin:0 0 24px}
#search{width:100%;padding:8px 12px;background:var(--bg2);border:1px solid var(--border);
  border-radius:6px;color:var(--text);font-size:.9rem;outline:none}
#search:focus{border-color:var(--accent)}
.result{padding:8px 12px;border-radius:6px;cursor:pointer}
.result:hover{background:var(--bg2)}
.result .mod{color:var(--muted);font-size:.8rem}
"""

SEARCH_JS = """
<script>
const idx=__SEARCH_INDEX__;
const box=document.getElementById('search');
const res=document.getElementById('results');
if(box){
  box.addEventListener('input',()=>{
    const q=box.value.trim().toLowerCase();
    res.innerHTML='';
    if(!q||q.length<2)return;
    const hits=idx.filter(e=>e.name.toLowerCase().includes(q)||e.mod.toLowerCase().includes(q)).slice(0,20);
    hits.forEach(h=>{
      const d=document.createElement('div');
      d.className='result';
      d.innerHTML='<a href="'+h.url+'"><strong>'+h.name+'</strong><span class="mod"> — '+h.mod+'</span></a>';
      res.appendChild(d);
    });
  });
}
</script>
"""

def esc(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"','&quot;')

def highlight_sig(sig: str) -> str:
    """Syntax-highlight a function signature string."""
    kws = {"fn","pub","extern","let","mut","return","if","else","struct","enum","union",
           "import","from","module","as","shadow","assert","while","for","in"}
    types = {"int","float","bool","string","void","array","List","HashMap","Set","Result",
             "Option","StringBuilder"}
    # Tokenize roughly: word boundaries
    def repl(m):
        w = m.group(0)
        if w in kws: return f'<span class="kw">{esc(w)}</span>'
        if w in types or (w and w[0].isupper()): return f'<span class="tp">{esc(w)}</span>'
        return esc(w)
    return re.sub(r'[A-Za-z_][A-Za-z0-9_]*', repl, sig)

def render_fn_card(export: dict) -> str:
    name  = esc(export.get("name",""))
    sig   = export.get("signature","")
    is_pub = export.get("is_public", False)
    is_ext = export.get("is_extern", False)
    badges = ""
    if is_pub:  badges += '<span class="badge">pub</span>'
    if is_ext:  badges += '<span class="badge">extern</span>'
    return (
        f'<div class="fn-card" id="fn-{name}">'
        f'<div class="fn-sig">{highlight_sig(sig)}</div>'
        f'{badges}'
        f'</div>'
    )

def page(title: str, body: str, sidebar_html: str, search_index_json: str = "[]",
         extra_head: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{esc(title)} — nanolang stdlib</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style>{extra_head}
</head>
<body>
<header>
  <h1>📦 nanolang stdlib</h1>
  <span class="ver">API Reference</span>
</header>
<div class="layout">
  <nav class="sidebar">{sidebar_html}</nav>
  <main>{body}</main>
</div>
{SEARCH_JS.replace('__SEARCH_INDEX__', search_index_json)}
</body></html>"""

# ── Build sidebar ─────────────────────────────────────────────────────────────

def build_sidebar(modules: list[dict], active_mod: str = "", root_prefix: str = "") -> str:
    groups: dict[str, list] = {}
    for m in modules:
        grp = m["group"]
        groups.setdefault(grp, []).append(m)
    html = ""
    html += f'<a href="{root_prefix}index.html">🏠 Index</a>\n'
    for grp, mods in sorted(groups.items()):
        html += f'<div class="group">{esc(grp)}</div>\n'
        for m in sorted(mods, key=lambda x: x["name"]):
            active = ' class="active"' if m["name"] == active_mod else ''
            html += f'<a href="{root_prefix}{m["slug"]}/index.html"{active}>{esc(m["name"])}</a>\n'
    return html

# ── Per-module page ───────────────────────────────────────────────────────────

def build_module_page(mod: dict, all_modules: list[dict], out_dir: Path):
    mod_dir = out_dir / mod["slug"]
    mod_dir.mkdir(parents=True, exist_ok=True)

    exports = mod.get("exports", [])
    fns = [e for e in exports if e.get("kind") == "function"]
    structs = [e for e in exports if e.get("kind") in ("struct","type")]

    body = f'<h2>{esc(mod["name"])}</h2>\n'
    body += f'<p class="mod-desc">{esc(mod["desc"])}</p>\n' if mod["desc"] else ""
    body += f'<p class="mod-desc" style="color:var(--muted);font-size:.82rem">Source: <code>{esc(mod["path"])}</code></p>\n'

    if structs:
        body += "<h3>Types</h3>\n"
        for s in structs:
            body += render_fn_card(s)

    if fns:
        body += f"<h3>Functions ({len(fns)})</h3>\n"
        for fn in fns:
            body += render_fn_card(fn)
    else:
        body += "<p style='color:var(--muted)'>No exported functions found.</p>\n"

    sidebar = build_sidebar(all_modules, active_mod=mod["name"], root_prefix="../")
    html = page(mod["name"], body, sidebar)
    (mod_dir / "index.html").write_text(html, encoding="utf-8")

# ── Index page ────────────────────────────────────────────────────────────────

def build_index_page(all_modules: list[dict], search_index: list[dict], out_dir: Path):
    body = """
<h2>stdlib API Reference</h2>
<p class="mod-desc">All public nanolang standard library modules. Click a module to browse its API.</p>
<div class="search-wrap">
  <input id="search" type="search" placeholder="🔍 Search functions and types..." autocomplete="off">
  <div id="results"></div>
</div>
"""
    groups: dict[str, list] = {}
    for m in all_modules:
        groups.setdefault(m["group"], []).append(m)

    for grp in sorted(groups.keys()):
        body += f'<h3>{esc(grp)}</h3>\n'
        for m in sorted(groups[grp], key=lambda x: x["name"]):
            fn_count = len([e for e in m.get("exports",[]) if e.get("kind")=="function"])
            body += (
                f'<div class="fn-card">'
                f'<a href="{esc(m["slug"])}/index.html"><strong>{esc(m["name"])}</strong></a>'
                f'<span class="badge">{fn_count} fn</span>'
                f'<span class="mod-desc" style="display:inline;margin-left:12px;font-size:.85rem;color:var(--muted)">'
                f'{esc(m["desc"][:120])}</span>'
                f'</div>\n'
            )

    sidebar = build_sidebar(all_modules, root_prefix="")
    idx_json = json.dumps(search_index)
    html = page("Index", body, sidebar, search_index_json=idx_json)
    (out_dir / "index.html").write_text(html, encoding="utf-8")

# ── AgentFS publish ───────────────────────────────────────────────────────────

def publish_to_agentfs(out_dir: Path, base_url: str, verbose: bool):
    base_url = base_url.rstrip("/")
    prefix = f"{base_url}/docs/nanolang"
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(out_dir))
        url = f"{prefix}/{rel}"
        ct = "text/html" if path.suffix == ".html" else "application/json"
        try:
            data = path.read_bytes()
            req = urllib.request.Request(url, data=data, method="PUT",
                                          headers={"Content-Type": ct, "Content-Length": str(len(data))})
            with urllib.request.urlopen(req, timeout=10) as r:
                if verbose:
                    print(f"  PUT {url} → {r.status}")
        except Exception as e:
            print(f"  [warn] PUT {url} failed: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    nanoc = args.nanoc
    stdlib_dir = Path(args.stdlib_dir)
    out_dir = Path(args.out)
    verbose = args.verbose

    if not Path(nanoc).exists():
        sys.exit(f"error: nanoc not found at {nanoc}")

    nano_files = sorted(stdlib_dir.rglob("*.nano"))
    print(f"Reflecting {len(nano_files)} stdlib modules...")

    all_modules: list[dict] = []
    search_index: list[dict] = []

    for nano_path in nano_files:
        if verbose:
            print(f"  → {nano_path}")
        data = reflect_module(nanoc, str(nano_path), verbose)
        if data is None:
            continue

        # Compute group from directory structure
        rel = nano_path.relative_to(stdlib_dir)
        parts = list(rel.parts)
        grp = parts[0].replace("_", " ").title() if len(parts) > 1 else "Core"
        mod_name = data.get("module") or rel.stem
        slug = str(rel.parent / rel.stem).replace("/","_").replace("\\","_")

        exports = data.get("exports", [])
        desc = extract_module_doc(str(nano_path))

        mod_entry = {
            "name":    mod_name,
            "slug":    slug,
            "path":    str(nano_path),
            "group":   grp,
            "desc":    desc,
            "exports": exports,
        }
        all_modules.append(mod_entry)

        # Build search index entries
        for exp in exports:
            search_index.append({
                "name": exp.get("name",""),
                "mod":  mod_name,
                "kind": exp.get("kind","fn"),
                "url":  f"{slug}/index.html#fn-{exp.get('name','')}",
            })

    print(f"Building site for {len(all_modules)} modules ({len(search_index)} symbols)...")

    # Clean and recreate output dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Per-module pages
    for mod in all_modules:
        build_module_page(mod, all_modules, out_dir)

    # Index page
    build_index_page(all_modules, search_index, out_dir)

    # Search index JSON
    (out_dir / "search_index.json").write_text(
        json.dumps(search_index, indent=2), encoding="utf-8"
    )

    total_html = sum(1 for _ in out_dir.rglob("*.html"))
    print(f"✅ Site built: {total_html} HTML pages → {out_dir}/")
    print(f"   Open: {out_dir}/index.html")

    # Optional: publish to AgentFS
    if args.agentfs_url:
        print(f"Publishing to AgentFS {args.agentfs_url}...")
        publish_to_agentfs(out_dir, args.agentfs_url, verbose)
        print(f"   Live at: {args.agentfs_url}/docs/nanolang/index.html")

if __name__ == "__main__":
    main()
