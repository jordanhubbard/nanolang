#!/usr/bin/env bash
# nano-pkg.sh — NanoLang Package Manager
# Shell-based tooling layer for `nanoc pkg install` and `nanoc pkg publish`.
# Registry: Git-based (github.com/jordanhubbard/nano-packages)
# Manifest: nano.toml
# Lockfile: nano.lock (JSON)
#
# I keep things simple. Git is my registry. TOML is my manifest.
# Dependencies are resolved, pinned, and checksummed.

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

REGISTRY_URL="${NANO_REGISTRY:-https://github.com/jordanhubbard/nano-packages.git}"
REGISTRY_BRANCH="${NANO_REGISTRY_BRANCH:-main}"
CACHE_DIR="${NANO_PKG_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/nanolang/packages}"
MANIFEST="nano.toml"
LOCKFILE="nano.lock"
MODULES_DIR="modules"
VERBOSE="${NANO_VERBOSE_BUILD:-0}"

# ============================================================================
# Utilities
# ============================================================================

die() { printf "error: %s\n" "$*" >&2; exit 1; }
info() { printf "  %s\n" "$*"; }
verbose() { [[ "$VERBOSE" == "1" ]] && printf "  [verbose] %s\n" "$*" || true; }

# Simple TOML parser — extracts key = "value" pairs from a section.
# Usage: toml_get <file> <section> <key>
# Handles [section] headers and bare key = "value" or key = 'value' lines.
toml_get() {
    local file="$1" section="$2" key="$3"
    awk -v section="$section" -v key="$key" '
        /^\[/ {
            gsub(/^\[[ \t]*|[ \t]*\]$/, "")
            current_section = $0
            next
        }
        current_section == section {
            split($0, parts, "=")
            k = parts[1]
            gsub(/^[ \t]+|[ \t]+$/, "", k)
            if (k == key) {
                v = substr($0, index($0, "=") + 1)
                gsub(/^[ \t]+|[ \t]+$/, "", v)
                gsub(/^["'"'"']|["'"'"']$/, "", v)
                print v
                exit
            }
        }
    ' "$file"
}

# Extract an array value from TOML (simple single-line arrays only)
# Usage: toml_get_array <file> <section> <key>
# Returns one item per line
toml_get_array() {
    local file="$1" section="$2" key="$3"
    awk -v section="$section" -v key="$key" '
        /^\[/ {
            gsub(/^\[[ \t]*|[ \t]*\]$/, "")
            current_section = $0
            next
        }
        current_section == section {
            split($0, parts, "=")
            k = parts[1]
            gsub(/^[ \t]+|[ \t]+$/, "", k)
            if (k == key) {
                v = substr($0, index($0, "=") + 1)
                gsub(/^[ \t]+|[ \t]+$/, "", v)
                gsub(/^\[|\]$/, "", v)
                n = split(v, items, ",")
                for (i = 1; i <= n; i++) {
                    item = items[i]
                    gsub(/^[ \t]+|[ \t]+$/, "", item)
                    gsub(/^["'"'"']|["'"'"']$/, "", item)
                    if (item != "") print item
                }
                exit
            }
        }
    ' "$file"
}

# Parse [dependencies] section: returns "name version_constraint" lines
# Format: name = ">=1.0.0" or name = "1.2.3"
toml_get_deps() {
    local file="$1"
    awk '
        /^\[dependencies\]/ { in_deps = 1; next }
        /^\[/ { in_deps = 0; next }
        in_deps && /=/ {
            split($0, parts, "=")
            k = parts[1]
            gsub(/^[ \t]+|[ \t]+$/, "", k)
            v = substr($0, index($0, "=") + 1)
            gsub(/^[ \t]+|[ \t]+$/, "", v)
            gsub(/^["'"'"']|["'"'"']$/, "", v)
            if (k != "" && v != "") print k " " v
        }
    ' "$file"
}

# Semantic version comparison: returns 0 if ver satisfies constraint
# Supports: "1.2.3" (exact), ">=1.2.0", "^1.2.0" (compatible), "*" (any)
semver_satisfies() {
    local ver="$1" constraint="$2"

    # Wildcard
    [[ "$constraint" == "*" ]] && return 0

    # Strip leading operators
    local op="="
    local cver="$constraint"
    if [[ "$constraint" == ">="* ]]; then
        op=">="
        cver="${constraint:2}"
    elif [[ "$constraint" == "^"* ]]; then
        op="^"
        cver="${constraint:1}"
    elif [[ "$constraint" == "~"* ]]; then
        op="~"
        cver="${constraint:1}"
    fi

    # Parse versions into major.minor.patch
    IFS='.' read -r v_maj v_min v_pat <<< "$ver"
    IFS='.' read -r c_maj c_min c_pat <<< "$cver"
    v_maj=${v_maj:-0}; v_min=${v_min:-0}; v_pat=${v_pat:-0}
    c_maj=${c_maj:-0}; c_min=${c_min:-0}; c_pat=${c_pat:-0}

    case "$op" in
        "=")
            [[ "$v_maj" -eq "$c_maj" && "$v_min" -eq "$c_min" && "$v_pat" -eq "$c_pat" ]]
            ;;
        ">=")
            if [[ "$v_maj" -gt "$c_maj" ]]; then return 0; fi
            if [[ "$v_maj" -lt "$c_maj" ]]; then return 1; fi
            if [[ "$v_min" -gt "$c_min" ]]; then return 0; fi
            if [[ "$v_min" -lt "$c_min" ]]; then return 1; fi
            [[ "$v_pat" -ge "$c_pat" ]]
            ;;
        "^")
            # Compatible: same major, >= minor.patch
            [[ "$v_maj" -eq "$c_maj" ]] || return 1
            if [[ "$v_min" -gt "$c_min" ]]; then return 0; fi
            if [[ "$v_min" -lt "$c_min" ]]; then return 1; fi
            [[ "$v_pat" -ge "$c_pat" ]]
            ;;
        "~")
            # Approximately: same major.minor, >= patch
            [[ "$v_maj" -eq "$c_maj" && "$v_min" -eq "$c_min" ]] || return 1
            [[ "$v_pat" -ge "$c_pat" ]]
            ;;
    esac
}

# Compute SHA-256 checksum of a directory (deterministic)
dir_checksum() {
    local dir="$1"
    find "$dir" -type f -not -path '*/.git/*' -print0 | sort -z | \
        xargs -0 sha256sum 2>/dev/null | sha256sum | awk '{print $1}'
}

# ============================================================================
# Registry Operations
# ============================================================================

# Ensure the local registry cache is up to date
registry_sync() {
    mkdir -p "$CACHE_DIR"
    local registry_dir="$CACHE_DIR/registry"

    if [[ -d "$registry_dir/.git" ]]; then
        verbose "Updating registry cache..."
        (cd "$registry_dir" && git fetch --quiet origin && git reset --quiet --hard "origin/$REGISTRY_BRANCH") || \
            die "Failed to update registry. Check your network connection."
    else
        verbose "Cloning registry from $REGISTRY_URL..."
        git clone --quiet --branch "$REGISTRY_BRANCH" --depth 1 "$REGISTRY_URL" "$registry_dir" || \
            die "Failed to clone registry from $REGISTRY_URL"
    fi
}

# List available versions of a package in the registry
# Registry layout: packages/<name>/<version>/
registry_versions() {
    local name="$1"
    local registry_dir="$CACHE_DIR/registry"
    local pkg_dir="$registry_dir/packages/$name"

    if [[ ! -d "$pkg_dir" ]]; then
        return 1
    fi

    # List version directories, sorted by semver (newest first)
    find "$pkg_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | \
        sort -t. -k1,1nr -k2,2nr -k3,3nr
}

# Resolve the best matching version for a constraint
registry_resolve() {
    local name="$1" constraint="$2"

    local versions
    versions=$(registry_versions "$name") || die "Package '$name' not found in registry"

    while IFS= read -r ver; do
        if semver_satisfies "$ver" "$constraint"; then
            echo "$ver"
            return 0
        fi
    done <<< "$versions"

    die "No version of '$name' satisfies constraint '$constraint'"
}

# ============================================================================
# Lockfile Operations
# ============================================================================

# Initialize an empty lockfile
lockfile_init() {
    cat > "$LOCKFILE" <<'EOF'
{
  "version": 1,
  "packages": {}
}
EOF
}

# Read a locked version for a package (empty if not locked)
lockfile_get_version() {
    local name="$1"
    if [[ ! -f "$LOCKFILE" ]]; then
        return
    fi
    # Use python3 or jq if available, fall back to grep
    if command -v jq &>/dev/null; then
        jq -r ".packages[\"$name\"].version // empty" "$LOCKFILE" 2>/dev/null
    elif command -v python3 &>/dev/null; then
        python3 -c "
import json, sys
with open('$LOCKFILE') as f:
    data = json.load(f)
pkg = data.get('packages', {}).get('$name', {})
v = pkg.get('version', '')
if v: print(v)
" 2>/dev/null
    else
        grep -A2 "\"$name\"" "$LOCKFILE" 2>/dev/null | grep '"version"' | \
            sed 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/'
    fi
}

# Write/update a package entry in the lockfile
lockfile_set() {
    local name="$1" version="$2" checksum="$3" source="$4"
    if command -v python3 &>/dev/null; then
        python3 -c "
import json, sys
try:
    with open('$LOCKFILE') as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {'version': 1, 'packages': {}}
data['packages']['$name'] = {
    'version': '$version',
    'checksum': '$checksum',
    'source': '$source'
}
with open('$LOCKFILE', 'w') as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write('\n')
"
    else
        die "python3 is required for lockfile management"
    fi
}

# ============================================================================
# Install Command
# ============================================================================

cmd_install() {
    local specific_pkg="${1:-}"

    if [[ ! -f "$MANIFEST" ]]; then
        die "No $MANIFEST found in current directory. Run 'nanoc-pkg init' to create one."
    fi

    # Sync registry
    printf "Syncing package registry...\n"
    registry_sync

    # Initialize lockfile if missing
    if [[ ! -f "$LOCKFILE" ]]; then
        lockfile_init
    fi

    # Create modules directory
    mkdir -p "$MODULES_DIR"

    # Collect dependencies
    local deps
    if [[ -n "$specific_pkg" ]]; then
        # Install a specific package (use * for any version)
        deps="$specific_pkg *"
    else
        deps=$(toml_get_deps "$MANIFEST")
    fi

    if [[ -z "$deps" ]]; then
        info "No dependencies declared in $MANIFEST."
        return 0
    fi

    local installed=0
    local up_to_date=0
    local failed=0

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local name version_constraint
        name=$(echo "$line" | awk '{print $1}')
        version_constraint=$(echo "$line" | awk '{print $2}')
        version_constraint="${version_constraint:-*}"

        verbose "Resolving $name $version_constraint..."

        # Check lockfile first
        local locked_ver
        locked_ver=$(lockfile_get_version "$name")

        local resolved_ver
        if [[ -n "$locked_ver" ]] && semver_satisfies "$locked_ver" "$version_constraint"; then
            resolved_ver="$locked_ver"
            verbose "Using locked version $locked_ver for $name"
        else
            resolved_ver=$(registry_resolve "$name" "$version_constraint")
            verbose "Resolved $name to $resolved_ver"
        fi

        # Check if already installed at correct version
        local target_dir="$MODULES_DIR/$name"
        local pkg_source="$CACHE_DIR/registry/packages/$name/$resolved_ver"

        if [[ -d "$target_dir" ]]; then
            local current_checksum
            current_checksum=$(dir_checksum "$target_dir")
            local locked_checksum
            locked_checksum=$(lockfile_get_version "$name" 2>/dev/null && \
                jq -r ".packages[\"$name\"].checksum // empty" "$LOCKFILE" 2>/dev/null || true)

            if [[ "$current_checksum" == "$locked_checksum" ]] && \
               [[ "$(lockfile_get_version "$name")" == "$resolved_ver" ]]; then
                verbose "$name@$resolved_ver is up to date"
                ((up_to_date++)) || true
                continue
            fi
        fi

        # Copy package from registry cache to modules/
        if [[ ! -d "$pkg_source" ]]; then
            printf "  ✗ %s@%s — not found in registry cache\n" "$name" "$resolved_ver"
            ((failed++)) || true
            continue
        fi

        verbose "Installing $name@$resolved_ver..."
        rm -rf "$target_dir"
        cp -r "$pkg_source" "$target_dir"

        # Compute checksum of installed package
        local checksum
        checksum=$(dir_checksum "$target_dir")

        # Update lockfile
        lockfile_set "$name" "$resolved_ver" "$checksum" "registry"

        info "✓ $name@$resolved_ver"
        ((installed++)) || true

        # Recursively install sub-dependencies if the package has its own nano.toml
        if [[ -f "$target_dir/$MANIFEST" ]]; then
            local subdeps
            subdeps=$(toml_get_deps "$target_dir/$MANIFEST")
            if [[ -n "$subdeps" ]]; then
                verbose "Processing sub-dependencies of $name..."
                while IFS= read -r subline; do
                    [[ -z "$subline" ]] && continue
                    local subname subconstraint
                    subname=$(echo "$subline" | awk '{print $1}')
                    subconstraint=$(echo "$subline" | awk '{print $2}')
                    # Append to our deps list for processing
                    deps="$deps
$subname $subconstraint"
                done <<< "$subdeps"
            fi
        fi

    done <<< "$deps"

    printf "\nDone. %d installed, %d up to date" "$installed" "$up_to_date"
    if [[ "$failed" -gt 0 ]]; then
        printf ", %d failed" "$failed"
    fi
    printf ".\n"

    if [[ "$failed" -gt 0 ]]; then
        return 1
    fi
}

# ============================================================================
# Publish Command
# ============================================================================

cmd_publish() {
    if [[ ! -f "$MANIFEST" ]]; then
        die "No $MANIFEST found. Cannot publish without a package manifest."
    fi

    local name version description
    name=$(toml_get "$MANIFEST" "package" "name")
    version=$(toml_get "$MANIFEST" "package" "version")
    description=$(toml_get "$MANIFEST" "package" "description")

    [[ -z "$name" ]] && die "Missing 'name' in [package] section of $MANIFEST"
    [[ -z "$version" ]] && die "Missing 'version' in [package] section of $MANIFEST"

    # Validate version format (semver)
    if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
        die "Version '$version' is not valid semver (expected: MAJOR.MINOR.PATCH)"
    fi

    printf "Publishing %s@%s...\n" "$name" "$version"

    # Sync registry
    registry_sync
    local registry_dir="$CACHE_DIR/registry"

    # Check if version already exists
    local target="$registry_dir/packages/$name/$version"
    if [[ -d "$target" ]]; then
        die "Version $version of $name already exists in the registry. Bump your version."
    fi

    # Build the package directory
    mkdir -p "$target"

    # Determine which files to include
    local include_patterns=()
    include_patterns+=("*.nano")
    include_patterns+=("*.c")
    include_patterns+=("*.h")
    include_patterns+=("module.json")
    include_patterns+=("$MANIFEST")
    include_patterns+=("README.md")
    include_patterns+=("LICENSE")

    # Copy matching files
    local file_count=0
    for pattern in "${include_patterns[@]}"; do
        for f in $pattern; do
            [[ -f "$f" ]] || continue
            cp "$f" "$target/"
            ((file_count++)) || true
        done
    done

    if [[ "$file_count" -eq 0 ]]; then
        rm -rf "$target"
        die "No publishable files found. Expected .nano, .c, .h, module.json, etc."
    fi

    # Generate a package index entry
    local checksum
    checksum=$(dir_checksum "$target")

    # Create/update the package metadata in the registry
    local meta_file="$registry_dir/packages/$name/package.json"
    if command -v python3 &>/dev/null; then
        python3 -c "
import json, os
meta_path = '$meta_file'
try:
    with open(meta_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {'name': '$name', 'description': '$description', 'versions': {}}
data['description'] = '$description' or data.get('description', '')
data['versions']['$version'] = {
    'checksum': '$checksum',
    'files': $file_count
}
os.makedirs(os.path.dirname(meta_path), exist_ok=True)
with open(meta_path, 'w') as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write('\n')
"
    fi

    # Stage and commit to the local registry clone
    (
        cd "$registry_dir"
        git add "packages/$name/"
        git commit --quiet -m "Publish $name@$version" || true
    )

    info "✓ $name@$version staged in local registry"
    info ""
    info "To push to the remote registry:"
    info "  cd $registry_dir"
    info "  git push origin $REGISTRY_BRANCH"
    info ""
    info "Package contents ($file_count files):"
    ls -1 "$target/" | sed 's/^/    /'
}

# ============================================================================
# Init Command
# ============================================================================

cmd_init() {
    if [[ -f "$MANIFEST" ]]; then
        die "$MANIFEST already exists. Edit it directly."
    fi

    local dir_name
    dir_name=$(basename "$(pwd)")

    cat > "$MANIFEST" <<EOF
[package]
name = "$dir_name"
version = "0.1.0"
description = ""
authors = []
license = "MIT"

[dependencies]
# Add dependencies here:
# some_package = ">=1.0.0"
# another_pkg = "^2.1.0"
EOF

    info "✓ Created $MANIFEST"
    info "  Edit it to add your package metadata and dependencies."
}

# ============================================================================
# Info / List Commands
# ============================================================================

cmd_list() {
    if [[ ! -f "$LOCKFILE" ]]; then
        info "No $LOCKFILE found. No packages installed."
        return
    fi

    printf "Installed packages (from %s):\n" "$LOCKFILE"
    if command -v python3 &>/dev/null; then
        python3 -c "
import json
with open('$LOCKFILE') as f:
    data = json.load(f)
for name, info in sorted(data.get('packages', {}).items()):
    print(f\"  {name} {info['version']}  ({info.get('checksum', '?')[:12]}...)\")
"
    else
        cat "$LOCKFILE"
    fi
}

cmd_info() {
    local name="${1:-}"
    [[ -z "$name" ]] && die "Usage: nanoc-pkg info <package-name>"

    registry_sync

    local registry_dir="$CACHE_DIR/registry"
    local meta_file="$registry_dir/packages/$name/package.json"

    if [[ ! -f "$meta_file" ]]; then
        die "Package '$name' not found in registry."
    fi

    if command -v python3 &>/dev/null; then
        python3 -c "
import json
with open('$meta_file') as f:
    data = json.load(f)
print(f\"Package: {data['name']}\")
print(f\"Description: {data.get('description', '(none)')}\")
print(f\"Versions: {', '.join(sorted(data.get('versions', {}).keys(), reverse=True))}\")
"
    else
        cat "$meta_file"
    fi
}

# ============================================================================
# Update Command — re-resolve all deps, ignoring lockfile
# ============================================================================

cmd_update() {
    if [[ ! -f "$MANIFEST" ]]; then
        die "No $MANIFEST found."
    fi

    printf "Updating all packages to latest compatible versions...\n"

    # Remove lockfile to force re-resolution
    rm -f "$LOCKFILE"
    cmd_install
}

# ============================================================================
# Main Dispatch
# ============================================================================

usage() {
    cat <<'EOF'
nanoc-pkg — NanoLang Package Manager

Usage:
  nanoc-pkg install [<package>]    Install dependencies from nano.toml (or a specific package)
  nanoc-pkg publish                Publish current package to the registry
  nanoc-pkg update                 Update all packages to latest compatible versions
  nanoc-pkg init                   Create a new nano.toml manifest
  nanoc-pkg list                   List installed packages from nano.lock
  nanoc-pkg info <package>         Show info about a registry package

Environment:
  NANO_REGISTRY          Git URL for the package registry
  NANO_REGISTRY_BRANCH   Branch to use (default: main)
  NANO_PKG_CACHE         Cache directory (default: ~/.cache/nanolang/packages)
  NANO_VERBOSE_BUILD     Set to 1 for verbose output

Files:
  nano.toml              Package manifest (name, version, dependencies)
  nano.lock              Lockfile (pinned versions + checksums)

EOF
}

main() {
    local cmd="${1:-}"
    shift || true

    case "$cmd" in
        install)  cmd_install "$@" ;;
        publish)  cmd_publish "$@" ;;
        update)   cmd_update "$@" ;;
        init)     cmd_init "$@" ;;
        list)     cmd_list "$@" ;;
        info)     cmd_info "$@" ;;
        help|--help|-h|"")  usage ;;
        *)        die "Unknown command: $cmd. Run 'nanoc-pkg help' for usage." ;;
    esac
}

main "$@"
