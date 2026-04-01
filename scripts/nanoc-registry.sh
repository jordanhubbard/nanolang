#!/usr/bin/env bash
# nanoc-registry.sh — nanolang package registry client
#
# Subcommands:
#   nanoc pkg search <query>            — search the registry
#   nanoc pkg info <name>               — show package info + versions
#   nanoc pkg install [<name>@<ver>]    — install packages (from nano.toml deps if no arg)
#   nanoc pkg publish                   — publish current package to registry
#   nanoc pkg lock                      — regenerate nano.lock without installing
#   nanoc pkg list                      — list installed packages in this project
#
# Config (env vars):
#   NANO_REGISTRY   — registry base URL (default: http://localhost:3900)
#   NANO_PKG_CACHE  — cache dir (default: ~/.cache/nanolang/packages)
#   NANO_SIGN_KEY   — Ed25519 keypair file for publish (default: ~/.nanoc/signing.key)
#   REGISTRY_TOKEN  — auth token for publish
#
# Manifest: nano.toml  (must have [package] section)
# Lockfile:  nano.lock  (JSON, pinned resolved versions + checksums)
# Modules:   nano_packages/<name>/ (installed modules)

set -euo pipefail

REGISTRY="${NANO_REGISTRY:-http://localhost:3900}"
CACHE_DIR="${NANO_PKG_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/nanolang/packages}"
SIGN_KEY="${NANO_SIGN_KEY:-$HOME/.nanoc/signing.key}"
MANIFEST="nano.toml"
LOCKFILE="nano.lock"
MODULES_DIR="nano_packages"
TOKEN="${REGISTRY_TOKEN:-}"

die()     { printf '\e[31merror:\e[0m %s\n' "$*" >&2; exit 1; }
info()    { printf '  \e[36m%s\e[0m\n' "$*"; }
success() { printf '  \e[32m✓\e[0m %s\n' "$*"; }
warn()    { printf '  \e[33mwarn:\e[0m %s\n' "$*" >&2; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Required tool not found: $1"; }
need_cmd curl
need_cmd python3

# ─── TOML helpers (pure bash/awk) ────────────────────────────────────────────

toml_get() {
    local file="$1" section="$2" key="$3"
    awk -v sec="$section" -v k="$key" '
        /^\[/ { gsub(/^\[[ \t]*|[ \t]*\]$/, ""); cur=$0; next }
        cur==sec {
            n=index($0,"="); if(!n) next
            lk=substr($0,1,n-1); gsub(/^[ \t]+|[ \t]+$/,"",lk)
            if(lk==k){ v=substr($0,n+1); gsub(/^[ \t]+|[ \t]+$/,"",v); gsub(/^["'"'"']|["'"'"']$/,"",v); print v; exit }
        }
    ' "$file"
}

toml_get_deps() {
    # Extract [deps] section: print lines as "name version"
    local file="$1"
    awk '
        /^\[/ { gsub(/^\[[ \t]*|[ \t]*\]$/, ""); cur=$0; next }
        cur=="deps" {
            n=index($0,"="); if(!n) next
            k=substr($0,1,n-1); gsub(/^[ \t]+|[ \t]+$/,"",k)
            v=substr($0,n+1); gsub(/^[ \t]+|[ \t]+$/,"",v); gsub(/^["'"'"']|["'"'"']$/,"",v)
            if(k!="") print k " " v
        }
    ' "$file"
}

# ─── JSON helpers (python3) ──────────────────────────────────────────────────

json_get() {
    # json_get <file> <jq-like path e.g. .versions.latest>
    python3 -c "
import sys, json
d=json.load(open('$1'))
for k in '$2'.strip('.').split('.'):
    if k: d=d[k]
print(d)" 2>/dev/null || echo ""
}

curl_json() {
    local url="$1"; shift
    local args=(-sf --connect-timeout 10)
    [[ -n "$TOKEN" ]] && args+=(-H "Authorization: Bearer $TOKEN")
    curl "${args[@]}" "$@" "$url" 2>/dev/null
}

sha256sum_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# ─── Lockfile ────────────────────────────────────────────────────────────────

lock_read() {
    [[ -f "$LOCKFILE" ]] && cat "$LOCKFILE" || echo '{}'
}

lock_write() {
    python3 -c "
import sys, json
data = json.loads(sys.stdin.read())
with open('$LOCKFILE','w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
}

lock_set() {
    # lock_set <name> <version> <sha256> <registry_url>
    local name="$1" version="$2" sha256="$3" registry="$4"
    python3 -c "
import sys, json
lock = json.loads(open('$LOCKFILE').read()) if __import__('os').path.exists('$LOCKFILE') else {}
lock['$name'] = {'version': '$version', 'sha256': '$sha256', 'registry': '$registry', 'resolved_at': __import__('datetime').datetime.utcnow().isoformat()+'Z'}
with open('$LOCKFILE','w') as f:
    json.dump(lock, f, indent=2)
    f.write('\n')
"
}

# ─── Subcommand: search ──────────────────────────────────────────────────────

cmd_search() {
    local query="${1:-}"
    local url="$REGISTRY/search"
    [[ -n "$query" ]] && url="$REGISTRY/search?q=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$query")"
    local result; result=$(curl_json "$url") || die "Registry unreachable: $REGISTRY"

    python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
pkgs = data.get('packages', [])
if not pkgs:
    print('  No packages found.')
    sys.exit(0)
print(f\"  {'NAME':<30} {'VERSION':<12} {'DESCRIPTION'}\")
print('  ' + '-'*72)
for p in pkgs:
    print(f\"  {p['name']:<30} {p.get('latest','?'):<12} {p.get('description','')[:40]}\")
print(f\"\n  {data.get('count',0)} package(s) found.\")
" <<< "$result"
}

# ─── Subcommand: info ────────────────────────────────────────────────────────

cmd_info() {
    local name="${1:-}"; [[ -z "$name" ]] && die "Usage: nanoc pkg info <package-name>"
    local result; result=$(curl_json "$REGISTRY/packages/$name") || die "Registry unreachable"
    python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
if 'error' in data:
    print(f'  error: {data[\"error\"]}'); sys.exit(1)
m = data.get('manifests', {}).get(data.get('latest',''), {})
print(f'  Name:        {data[\"name\"]}')
print(f'  Latest:      {data.get(\"latest\",\"?\")}')
print(f'  Description: {m.get(\"description\",\"(none)\")}')
print(f'  Author:      {m.get(\"author\",\"?\")}')
print(f'  License:     {m.get(\"license\",\"?\")}')
print(f'  Tags:        {\" \".join(m.get(\"tags\",[]))}')
print(f'  Versions:    {\" \".join(data.get(\"versions\",[]))}')
deps = m.get('deps', {})
if deps:
    print(f'  Deps:')
    for k,v in deps.items(): print(f'    {k}: {v}')
" <<< "$result"
}

# ─── Install a single package ────────────────────────────────────────────────

install_one() {
    local spec="$1"
    local name version_range
    if [[ "$spec" == *@* ]]; then
        name="${spec%%@*}"
        version_range="${spec##*@}"
    else
        name="$spec"
        version_range="*"
    fi

    info "Resolving $name@$version_range …"
    local meta; meta=$(curl_json "$REGISTRY/packages/$name/$version_range") || die "Registry unreachable"
    local err; err=$(echo "$meta" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('error',''))" 2>/dev/null)
    [[ -n "$err" ]] && die "$err"

    local resolved_version; resolved_version=$(echo "$meta" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('resolved_version',''))")
    [[ -z "$resolved_version" ]] && die "Could not resolve version for $name@$version_range"

    local sha256_expected; sha256_expected=$(echo "$meta" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('manifest',{}).get('sha256',''))")

    info "Installing $name@$resolved_version …"

    # Check cache first
    local cache_tar="$CACHE_DIR/$name/$resolved_version/${name}-${resolved_version}.tar.gz"
    if [[ ! -f "$cache_tar" ]]; then
        mkdir -p "$(dirname "$cache_tar")"
        local down_url="$REGISTRY/packages/$name/$resolved_version/tarball"
        curl -sf --connect-timeout 30 -o "$cache_tar" "$down_url" || die "Failed to download $name@$resolved_version"
    fi

    # Verify checksum if provided
    if [[ -n "$sha256_expected" ]]; then
        local actual; actual=$(sha256sum_file "$cache_tar")
        [[ "$actual" == "$sha256_expected" ]] || die "Checksum mismatch for $name@$resolved_version (expected $sha256_expected, got $actual)"
    fi

    # Extract to nano_packages/
    local dest="$MODULES_DIR/$name"
    rm -rf "$dest"
    mkdir -p "$dest"
    tar -xzf "$cache_tar" -C "$dest" --strip-components=1 2>/dev/null \
        || tar -xzf "$cache_tar" -C "$dest" 2>/dev/null \
        || die "Failed to extract tarball for $name"

    # Update lockfile
    local actual_sha; actual_sha=$(sha256sum_file "$cache_tar")
    lock_set "$name" "$resolved_version" "$actual_sha" "$REGISTRY"

    success "$name@$resolved_version installed to $dest"
}

# ─── Subcommand: install ────────────────────────────────────────────────────

cmd_install() {
    mkdir -p "$MODULES_DIR" "$CACHE_DIR"

    if [[ $# -gt 0 ]]; then
        # Install specific packages
        for spec in "$@"; do
            install_one "$spec"
        done
    else
        # Install from nano.toml [deps]
        [[ -f "$MANIFEST" ]] || die "No $MANIFEST found. Run 'nanoc pkg install <name>' to add a package."
        local deps; deps=$(toml_get_deps "$MANIFEST")
        if [[ -z "$deps" ]]; then
            info "No [deps] section in $MANIFEST — nothing to install."
            return 0
        fi
        while IFS=' ' read -r name version_range; do
            [[ -z "$name" ]] && continue
            install_one "${name}@${version_range}"
        done <<< "$deps"
    fi

    success "nano.lock updated."
}

# ─── Subcommand: publish ─────────────────────────────────────────────────────

cmd_publish() {
    [[ -f "$MANIFEST" ]] || die "No $MANIFEST found in current directory."
    [[ -n "$TOKEN" ]] || warn "REGISTRY_TOKEN not set — publish may fail if registry requires auth."

    local name; name=$(toml_get "$MANIFEST" package name)
    local version; version=$(toml_get "$MANIFEST" package version)
    local description; description=$(toml_get "$MANIFEST" package description)
    local author; author=$(toml_get "$MANIFEST" package author)
    local license; license=$(toml_get "$MANIFEST" package license)
    local main_file; main_file=$(toml_get "$MANIFEST" package main)
    [[ -z "$name" ]] && die "nano.toml missing [package].name"
    [[ -z "$version" ]] && die "nano.toml missing [package].version"

    info "Publishing $name@$version …"

    # Build tarball of source files
    local tmpdir; tmpdir=$(mktemp -d)
    local tarball="$tmpdir/${name}-${version}.tar.gz"

    # Package root = files listed in nano.toml [package].include, or all .nano + nano.toml
    local include_glob; include_glob=$(toml_get "$MANIFEST" package include)
    local src_dir="$tmpdir/src/${name}-${version}"
    mkdir -p "$src_dir"
    cp "$MANIFEST" "$src_dir/"
    if [[ -n "$include_glob" ]]; then
        # shellcheck disable=SC2086
        cp -r $include_glob "$src_dir/" 2>/dev/null || true
    else
        find . -maxdepth 3 -name "*.nano" -not -path "./$MODULES_DIR/*" -not -path "./.git/*" \
            -exec cp --parents {} "$src_dir/" \; 2>/dev/null || true
    fi

    tar -czf "$tarball" -C "$tmpdir/src" "${name}-${version}"

    # Sign with Ed25519 key (if nanoc sign is available)
    local pubkey_hex="" sig_hex=""
    if command -v nanoc >/dev/null 2>&1 && [[ -f "$SIGN_KEY" ]]; then
        info "Signing with $SIGN_KEY …"
        local sig_out; sig_out=$(nanoc sign "$tarball" --key "$SIGN_KEY" --output-hex 2>/dev/null || echo "")
        if [[ -n "$sig_out" ]]; then
            pubkey_hex=$(echo "$sig_out" | awk '/pubkey:/{print $2}')
            sig_hex=$(echo "$sig_out" | awk '/signature:/{print $2}')
        fi
    fi

    # Build manifest
    local manifest_json
    manifest_json=$(python3 -c "
import json, sys
m = {
    'name': sys.argv[1],
    'version': sys.argv[2],
    'description': sys.argv[3],
    'author': sys.argv[4],
    'license': sys.argv[5],
    'main': sys.argv[6] or 'lib.nano',
    'tags': [],
    'deps': {},
    'ed25519_pubkey': sys.argv[7],
    'ed25519_sig': sys.argv[8],
}
print(json.dumps(m))
" "$name" "$version" "$description" "$author" "$license" "$main_file" "$pubkey_hex" "$sig_hex")

    # Base64-encode tarball
    local tarball_b64; tarball_b64=$(base64 < "$tarball" | tr -d '\n')

    # POST to registry
    local payload; payload=$(python3 -c "
import json, sys
manifest = json.loads(sys.argv[1])
body = {'manifest': manifest, 'tarball_b64': sys.argv[2]}
print(json.dumps(body))
" "$manifest_json" "$tarball_b64")

    local result; result=$(curl_json "$REGISTRY/packages" -X POST \
        -H "Content-Type: application/json" \
        -d "$payload") || die "Failed to publish — registry unreachable"

    local err; err=$(echo "$result" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('error',''))" 2>/dev/null)
    [[ -n "$err" ]] && die "Registry error: $err"

    echo "$result" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print(f\"  published: {d['name']}@{d['version']} (sha256: {d['sha256'][:16]}…)\")
print(f\"  latest on registry: {d.get('latest','?')}\")
"

    rm -rf "$tmpdir"
    success "Published $name@$version to $REGISTRY"
}

# ─── Subcommand: list ────────────────────────────────────────────────────────

cmd_list() {
    [[ -f "$LOCKFILE" ]] || { info "No nano.lock — run 'nanoc pkg install'"; return; }
    python3 -c "
import json, sys
lock = json.loads(open('$LOCKFILE').read())
if not lock:
    print('  (no packages installed)')
    sys.exit(0)
print(f\"  {'PACKAGE':<30} {'VERSION':<12} {'SHA256'}\")
print('  ' + '-'*72)
for name, info in sorted(lock.items()):
    sha = info.get('sha256','?')[:16] + '…'
    print(f\"  {name:<30} {info.get('version','?'):<12} {sha}\")
"
}

# ─── Subcommand: lock (regenerate without re-downloading) ────────────────────

cmd_lock() {
    info "Regenerating $LOCKFILE from installed packages …"
    [[ -d "$MODULES_DIR" ]] || { warn "No $MODULES_DIR/ directory"; return; }
    local newlock="{}"
    for pkg_dir in "$MODULES_DIR"/*/; do
        local pkg_name; pkg_name=$(basename "$pkg_dir")
        local manifest_f="$pkg_dir/nano.toml"
        [[ -f "$manifest_f" ]] || continue
        local pkg_ver; pkg_ver=$(toml_get "$manifest_f" package version)
        [[ -z "$pkg_ver" ]] && continue
        # Find cached tarball for checksum
        local cached_tar="$CACHE_DIR/$pkg_name/$pkg_ver/${pkg_name}-${pkg_ver}.tar.gz"
        local sha=""
        [[ -f "$cached_tar" ]] && sha=$(sha256sum_file "$cached_tar")
        lock_set "$pkg_name" "$pkg_ver" "$sha" "$REGISTRY"
    done
    success "$LOCKFILE regenerated."
}

# ─── Main dispatch ────────────────────────────────────────────────────────────

subcmd="${1:-help}"
shift 2>/dev/null || true

case "$subcmd" in
    search)  cmd_search "$@" ;;
    info)    cmd_info "$@" ;;
    install) cmd_install "$@" ;;
    publish) cmd_publish "$@" ;;
    list)    cmd_list "$@" ;;
    lock)    cmd_lock "$@" ;;
    help|--help|-h)
        cat <<'EOF'
nanoc pkg — nano package manager

Usage:
  nanoc pkg search <query>        Search registry by name/description/tags
  nanoc pkg info <name>           Show package info and available versions
  nanoc pkg install               Install all [deps] from nano.toml
  nanoc pkg install <name>[@ver]  Install a specific package (semver range ok)
  nanoc pkg publish               Publish current package (requires nano.toml)
  nanoc pkg list                  List installed packages (from nano.lock)
  nanoc pkg lock                  Regenerate nano.lock from installed packages

Environment:
  NANO_REGISTRY=http://...   Registry URL (default: http://localhost:3900)
  NANO_PKG_CACHE=~/.cache/…  Package cache directory
  REGISTRY_TOKEN=<token>     Auth token for publishing
  NANO_SIGN_KEY=~/.nanoc/… Ed25519 signing keypair for publish
EOF
        ;;
    *) die "Unknown subcommand: $subcmd. Try 'nanoc pkg help'." ;;
esac
