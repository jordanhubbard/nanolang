#!/usr/bin/env bash
# nanoc-install — Install nano packages from the registry
#
# Usage:
#   nanoc install <pkg>[@<semver-range>] [<pkg2> ...]
#   nanoc install               # install from nano.lock / nano.packages.json
#
# Options:
#   --registry <url>   Registry base URL (default: $NANO_REGISTRY or https://packages.nanolang.org)
#   --packages-dir <d> Install directory (default: ./nano_packages)
#   --save             Update nano.packages.json with resolved versions
#   --dry-run          Resolve and print, but do not download
#   --verbose          Show detailed output
#
# Lock file:
#   nano.lock — JSON file pinning resolved exact versions:
#     {
#       "lockfile_version": 1,
#       "packages": {
#         "gpu-math": "1.2.0",
#         "nano-core": "0.3.1"
#       },
#       "resolved": {
#         "gpu-math@1.2.0": {
#           "tarball": "https://registry/api/v1/packages/gpu-math/1.2.0/tarball",
#           "sha256": "abc..."
#         }
#       }
#     }
#
# Dependencies declared in nano.packages.json:
#   {
#     "name": "my-project",
#     "dependencies": {
#       "gpu-math": "^1.0.0",
#       "nano-core": ">=0.3"
#     }
#   }
#
# Environment:
#   NANO_REGISTRY         Registry base URL
#   NANO_PACKAGES_DIR     Install directory (default: ./nano_packages)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────── #
REGISTRY="${NANO_REGISTRY:-https://packages.nanolang.org}"
PACKAGES_DIR="${NANO_PACKAGES_DIR:-./nano_packages}"
LOCK_FILE="./nano.lock"
PKG_JSON="./nano.packages.json"
DRY_RUN=false
VERBOSE=false
SAVE=false
PACKAGES=()

# ── Parse args ─────────────────────────────────────────────────────────── #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry)  REGISTRY="$2";      shift 2 ;;
    --packages-dir) PACKAGES_DIR="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=true;        shift ;;
    --verbose)   VERBOSE=true;        shift ;;
    --save)      SAVE=true;           shift ;;
    -*)          echo "Unknown flag: $1" >&2; exit 1 ;;
    *)           PACKAGES+=("$1");    shift ;;
  esac
done

log() { echo "[nanoc-install] $*"; }
verbose() { $VERBOSE && echo "[nanoc-install] $*" || true; }
die()  { echo "[nanoc-install] ERROR: $*" >&2; exit 1; }

# ── Semver resolve via registry ────────────────────────────────────────── #
resolve_version() {
  local name="$1" range="$2"
  local encoded_range
  encoded_range="$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$range" 2>/dev/null || echo "$range")"
  local url="${REGISTRY}/api/v1/resolve/${name}/${encoded_range}"
  verbose "Resolving: GET $url"
  local resp
  resp="$(curl -sf "$url" 2>/dev/null)" || die "Failed to resolve ${name}@${range} from $REGISTRY"
  python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['resolved'])" "$resp" 2>/dev/null \
    || die "Unexpected response resolving ${name}@${range}: $resp"
}

# ── Fetch and verify tarball ───────────────────────────────────────────── #
install_package() {
  local name="$1" version="$2"
  local tarball_url="${REGISTRY}/api/v1/packages/${name}/${version}/tarball"
  local dest="${PACKAGES_DIR}/${name}@${version}"

  if [[ -d "$dest" ]]; then
    verbose "Already installed: ${name}@${version} (${dest})"
    return 0
  fi

  log "Installing ${name}@${version}..."
  verbose "  Tarball: $tarball_url"

  if $DRY_RUN; then
    echo "  [dry-run] Would install ${name}@${version} → ${dest}"
    return 0
  fi

  mkdir -p "$dest"
  local tarball="${dest}/package.tar.gz"
  curl -sf -o "$tarball" "$tarball_url" || die "Failed to download ${name}@${version}"

  # Verify SHA-256 against registry manifest
  local manifest_url="${REGISTRY}/api/v1/packages/${name}/${version}"
  local manifest
  manifest="$(curl -sf "$manifest_url" 2>/dev/null)" || true
  local expected_sha
  expected_sha="$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('manifest',{}).get('_sha256',''))" "$manifest" 2>/dev/null)" || expected_sha=""
  if [[ -n "$expected_sha" ]]; then
    local actual_sha
    actual_sha="$(sha256sum "$tarball" | awk '{print $1}')"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
      rm -rf "$dest"
      die "SHA-256 mismatch for ${name}@${version}: expected $expected_sha, got $actual_sha"
    fi
    verbose "  SHA-256 verified: $actual_sha"
  fi

  # Unpack
  tar -xzf "$tarball" -C "$dest" 2>/dev/null || die "Failed to unpack ${name}@${version}"
  log "  ✓ Installed ${name}@${version} → ${dest}"

  # Write manifest for reference
  if [[ -n "$manifest" ]]; then
    echo "$manifest" | python3 -c "import json,sys; print(json.dumps(json.loads(sys.stdin.read()).get('manifest',{}),indent=2))" > "${dest}/nano.manifest.json" 2>/dev/null || true
  fi
}

# ── Update lock file ───────────────────────────────────────────────────── #
update_lockfile() {
  local -n pkg_map=$1  # nameref to associative array name→version
  local lock="{\"lockfile_version\":1,\"packages\":{"
  local first=true
  for name in "${!pkg_map[@]}"; do
    $first && first=false || lock+=","
    lock+="\"${name}\":\"${pkg_map[$name]}\""
  done
  lock+="}}"
  echo "$lock" | python3 -c "import json,sys; print(json.dumps(json.loads(sys.stdin.read()),indent=2))" > "$LOCK_FILE"
  log "Updated $LOCK_FILE"
}

# ── Main ───────────────────────────────────────────────────────────────── #
mkdir -p "$PACKAGES_DIR"

declare -A resolved_map

# If no packages specified: read from nano.packages.json + nano.lock
if [[ ${#PACKAGES[@]} -eq 0 ]]; then
  if [[ -f "$LOCK_FILE" ]]; then
    log "Using $LOCK_FILE (locked versions)"
    while IFS=":" read -r name version; do
      name="${name//[\" ]/}"
      version="${version//[\" ,]/}"
      [[ -z "$name" || -z "$version" || "$name" == "lockfile_version" || "$name" == "packages" ]] && continue
      resolved_map["$name"]="$version"
    done < <(python3 -c "
import json,sys
d=json.load(open('$LOCK_FILE'))
for k,v in d.get('packages',{}).items():
    print(f'{k}:{v}')
" 2>/dev/null)
  elif [[ -f "$PKG_JSON" ]]; then
    log "Reading dependencies from $PKG_JSON"
    while IFS="=" read -r name range; do
      [[ -z "$name" || -z "$range" ]] && continue
      version="$(resolve_version "$name" "$range")"
      resolved_map["$name"]="$version"
    done < <(python3 -c "
import json,sys
d=json.load(open('$PKG_JSON'))
for k,v in d.get('dependencies',{}).items():
    print(f'{k}={v}')
" 2>/dev/null)
  else
    die "No packages specified and no $PKG_JSON or $LOCK_FILE found"
  fi
else
  # Parse name[@range] args
  for pkg_arg in "${PACKAGES[@]}"; do
    if [[ "$pkg_arg" == *@* ]]; then
      name="${pkg_arg%%@*}"
      range="${pkg_arg#*@}"
    else
      name="$pkg_arg"
      range="latest"
    fi
    log "Resolving ${name}@${range}..."
    version="$(resolve_version "$name" "$range")"
    log "  Resolved: ${name}@${range} → ${version}"
    resolved_map["$name"]="$version"
  done
fi

# Install each resolved package
for name in "${!resolved_map[@]}"; do
  install_package "$name" "${resolved_map[$name]}"
done

# Optionally update lock file
if $SAVE || [[ ${#PACKAGES[@]} -gt 0 && ! -f "$LOCK_FILE" ]]; then
  update_lockfile resolved_map
fi

log "Done. ${#resolved_map[@]} package(s) installed to ${PACKAGES_DIR}/"
