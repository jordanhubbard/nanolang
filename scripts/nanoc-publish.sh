#!/usr/bin/env bash
# nanoc-publish — compile a .nano source file to WASM and publish to AgentFS
#
# Usage:
#   nanoc-publish <source.nano> [--agentfs <url>] [--capabilities <cap1,cap2,...>]
#                               [--dry-run] [--output <file.wasm>] [--verbose]
#
# Environment:
#   AGENTFS_URL   AgentFS base URL (default: http://sparky.tail407856.ts.net:8791)
#   AGENTFS_KEY   API key for AgentFS (optional, sent as Bearer token)
#
# Workflow:
#   1. Compile <source.nano> to WASM via `nanoc --target wasm`
#   2. Optionally inject agentos.capabilities custom section from annotations in source
#   3. POST binary to AgentFS /upload endpoint
#   4. Print content hash returned by AgentFS
#   5. (Optional) Trigger VibeEngine PROPOSE via RCC exec relay
#
# agentOS capability annotations in .nano source (comments parsed by this script):
#   @agentos.capabilities: fs,net
#   @agentos.name: my_agent
#   @agentos.version: 1.0.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOLANG_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────────────
AGENTFS_URL="${AGENTFS_URL:-http://sparky.tail407856.ts.net:8791}"
AGENTFS_KEY="${AGENTFS_KEY:-}"
VERBOSE=0
DRY_RUN=0
OUTPUT_WASM=""
CAPABILITIES=""
SOURCE_FILE=""
PROPOSE_TO_VIBE=0

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[publish]${NC} $*"; }
ok()      { echo -e "${GREEN}[publish]${NC} ✓ $*"; }
warn()    { echo -e "${YELLOW}[publish]${NC} ⚠ $*" >&2; }
err()     { echo -e "${RED}[publish]${NC} ✗ $*" >&2; }
verbose() { [[ $VERBOSE -eq 1 ]] && echo -e "${CYAN}[publish]${NC} $*" || true; }

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: nanoc-publish <source.nano> [options]

Compile a .nano source file to WASM and publish to AgentFS.

Options:
  --agentfs <url>         AgentFS base URL (default: \$AGENTFS_URL or sparky:8791)
  --capabilities <caps>   Comma-separated capability list (fs,net,gpu,...)
                          Overrides @agentos.capabilities annotation in source
  --output <file.wasm>    Save compiled WASM to this path (default: temp file)
  --propose               After upload, trigger VibeEngine PROPOSE via RCC exec
  --dry-run               Compile to WASM but do not upload
  --verbose               Verbose output
  -h, --help              Show this help

Environment variables:
  AGENTFS_URL   AgentFS base URL
  AGENTFS_KEY   Bearer token for AgentFS authentication

Capability annotations in .nano source (in comments):
  @agentos.capabilities: fs,net
  @agentos.name: my_agent
  @agentos.version: 1.0.0

Example:
  nanoc-publish src/my_agent.nano --capabilities fs,net
  AGENTFS_URL=http://localhost:8791 nanoc-publish examples/hello.nano --dry-run
EOF
    exit 0
}

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --agentfs) AGENTFS_URL="$2"; shift 2 ;;
        --capabilities) CAPABILITIES="$2"; shift 2 ;;
        --output) OUTPUT_WASM="$2"; shift 2 ;;
        --propose) PROPOSE_TO_VIBE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --verbose) VERBOSE=1; shift ;;
        -*)
            err "Unknown option: $1"
            echo "Use --help for usage." >&2
            exit 1
            ;;
        *)
            if [[ -z "$SOURCE_FILE" ]]; then
                SOURCE_FILE="$1"
            else
                err "Unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# ── Validate inputs ───────────────────────────────────────────────────────────
if [[ -z "$SOURCE_FILE" ]]; then
    err "No source file specified."
    echo "Usage: nanoc-publish <source.nano> [options]" >&2
    exit 1
fi

if [[ ! -f "$SOURCE_FILE" ]]; then
    err "Source file not found: $SOURCE_FILE"
    exit 1
fi

# ── Locate nanoc binary ───────────────────────────────────────────────────────
NANOC_BIN=""
for candidate in \
    "$NANOLANG_ROOT/bin/nanoc" \
    "$(command -v nanoc 2>/dev/null || true)"; do
    if [[ -x "$candidate" ]]; then
        NANOC_BIN="$candidate"
        break
    fi
done

if [[ -z "$NANOC_BIN" ]]; then
    err "nanoc binary not found. Build nanolang first: make -C '$NANOLANG_ROOT'"
    exit 1
fi
verbose "Using nanoc: $NANOC_BIN"

# ── Parse @agentos annotations from source ────────────────────────────────────
parse_annotation() {
    local file="$1" key="$2"
    grep -oP "(?<=@agentos\.$key:\s).*" "$file" 2>/dev/null | head -1 | xargs || true
}

ANNO_NAME="$(parse_annotation "$SOURCE_FILE" "name")"
ANNO_VERSION="$(parse_annotation "$SOURCE_FILE" "version")"
ANNO_CAPS="$(parse_annotation "$SOURCE_FILE" "capabilities")"

# CLI --capabilities overrides annotation
if [[ -z "$CAPABILITIES" && -n "$ANNO_CAPS" ]]; then
    CAPABILITIES="$ANNO_CAPS"
    verbose "Capabilities from annotation: $CAPABILITIES"
fi

AGENT_NAME="${ANNO_NAME:-$(basename "$SOURCE_FILE" .nano)}"
AGENT_VERSION="${ANNO_VERSION:-0.0.1}"

info "Publishing '$AGENT_NAME' v$AGENT_VERSION from $SOURCE_FILE"
[[ -n "$CAPABILITIES" ]] && info "Capabilities: $CAPABILITIES"

# ── Step 1: Compile to WASM ───────────────────────────────────────────────────
CLEANUP_WASM=0
if [[ -z "$OUTPUT_WASM" ]]; then
    OUTPUT_WASM="$(mktemp /tmp/nanoc-publish-XXXXXX.wasm)"
    CLEANUP_WASM=1
fi

info "Compiling $SOURCE_FILE → $OUTPUT_WASM"
if ! "$NANOC_BIN" "$SOURCE_FILE" --target wasm -o "$OUTPUT_WASM" ${VERBOSE:+--verbose}; then
    err "Compilation failed."
    [[ $CLEANUP_WASM -eq 1 ]] && rm -f "$OUTPUT_WASM"
    exit 1
fi

WASM_SIZE="$(wc -c < "$OUTPUT_WASM")"
ok "Compiled to WASM (${WASM_SIZE} bytes): $OUTPUT_WASM"

# ── Step 2: Inject agentos.capabilities custom WASM section ───────────────────
# The WASM custom section format: 0x00 (custom), len, namelen, "agentos.capabilities", data
# We append it as a raw binary section after the existing module.
if [[ -n "$CAPABILITIES" ]]; then
    INJECT_PY="$(mktemp /tmp/nanoc-cap-inject-XXXXXX.py)"
    cat > "$INJECT_PY" << 'PYEOF'
#!/usr/bin/env python3
"""
Inject an agentos.capabilities custom section into a WASM binary.
Custom section format (WASM spec §5.5.3):
  - section id: 0x00
  - section size: u32 (leb128)
  - name len: u32 (leb128)
  - name bytes: UTF-8
  - data bytes
"""
import sys, struct

def encode_leb128(n):
    result = b''
    while True:
        byte = n & 0x7f
        n >>= 7
        if n != 0:
            byte |= 0x80
        result += bytes([byte])
        if n == 0:
            break
    return result

wasm_file = sys.argv[1]
capabilities = sys.argv[2]  # e.g. "fs,net"
output_file = sys.argv[3]

with open(wasm_file, 'rb') as f:
    wasm = f.read()

# Validate WASM magic + version
if wasm[:4] != b'\x00asm':
    print(f"ERROR: Not a valid WASM file: {wasm_file}", file=sys.stderr)
    sys.exit(1)

# Build custom section
section_name = b'agentos.capabilities'
section_data = capabilities.encode('utf-8')

name_bytes = encode_leb128(len(section_name)) + section_name
content = name_bytes + section_data
section_body = b'\x00' + encode_leb128(len(content)) + content

# Append to WASM
with open(output_file, 'wb') as f:
    f.write(wasm + section_body)

print(f"Injected agentos.capabilities section: {capabilities}")
PYEOF
    if python3 "$INJECT_PY" "$OUTPUT_WASM" "$CAPABILITIES" "${OUTPUT_WASM}.tmp"; then
        mv "${OUTPUT_WASM}.tmp" "$OUTPUT_WASM"
        WASM_SIZE="$(wc -c < "$OUTPUT_WASM")"
        ok "Injected capabilities section (${WASM_SIZE} bytes total)"
    else
        warn "Failed to inject capabilities section — uploading without it"
        rm -f "${OUTPUT_WASM}.tmp"
    fi
    rm -f "$INJECT_PY"
fi

# ── Dry run exits here ─────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    ok "Dry run complete. WASM at: $OUTPUT_WASM"
    info "Would upload to: $AGENTFS_URL/upload"
    [[ $CLEANUP_WASM -eq 1 ]] && { info "Keeping WASM for inspection: $OUTPUT_WASM"; CLEANUP_WASM=0; }
    exit 0
fi

# ── Step 3: POST to AgentFS ───────────────────────────────────────────────────
info "Uploading to AgentFS: $AGENTFS_URL/upload"

# Build require param from capabilities
REQUIRE_PARAM=""
if [[ -n "$CAPABILITIES" ]]; then
    REQUIRE_PARAM="?require=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$CAPABILITIES" 2>/dev/null || echo "$CAPABILITIES")"
fi

AUTH_HEADER=""
if [[ -n "$AGENTFS_KEY" ]]; then
    AUTH_HEADER="-H \"Authorization: Bearer $AGENTFS_KEY\""
fi

UPLOAD_RESPONSE="$(curl -sf \
    --connect-timeout 10 \
    --max-time 30 \
    ${AGENTFS_KEY:+-H "Authorization: Bearer $AGENTFS_KEY"} \
    -X POST \
    -H "Content-Type: application/wasm" \
    -H "X-Agent-Name: $AGENT_NAME" \
    -H "X-Agent-Version: $AGENT_VERSION" \
    --data-binary "@$OUTPUT_WASM" \
    "${AGENTFS_URL}/upload${REQUIRE_PARAM}" 2>&1)" || {
    CURL_EXIT=$?
    err "AgentFS upload failed (curl exit $CURL_EXIT)."
    err "AgentFS URL: ${AGENTFS_URL}/upload${REQUIRE_PARAM}"
    err "Is AgentFS running and reachable? Try: curl -s ${AGENTFS_URL}/health"
    [[ $CLEANUP_WASM -eq 1 ]] && rm -f "$OUTPUT_WASM"
    exit 1
}

verbose "AgentFS response: $UPLOAD_RESPONSE"

# Parse content hash from response (AgentFS returns JSON with hash field)
CONTENT_HASH="$(echo "$UPLOAD_RESPONSE" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    h = d.get('hash') or d.get('content_hash') or d.get('id') or d.get('sha256') or '(unknown)'
    print(h)
except Exception as e:
    print('(parse error: ' + str(e) + ')')
" 2>/dev/null || echo "(unknown)")"

ok "Published to AgentFS!"
echo ""
echo "  Agent:   $AGENT_NAME v$AGENT_VERSION"
echo "  Hash:    $CONTENT_HASH"
[[ -n "$CAPABILITIES" ]] && echo "  Caps:    $CAPABILITIES"
echo "  Size:    ${WASM_SIZE} bytes"
echo ""
echo "  VibeEngine PROPOSE: nanoc-publish --propose (or POST /api/exec with PROPOSE hash)"

# ── Step 4: Optional VibeEngine PROPOSE via RCC exec relay ───────────────────
if [[ $PROPOSE_TO_VIBE -eq 1 ]]; then
    RCC_URL="${RCC_URL:-http://localhost:8789}"
    RCC_AGENT_TOKEN="${RCC_AGENT_TOKEN:-}"

    if [[ -z "$RCC_AGENT_TOKEN" ]] && [[ -f "$HOME/.rcc/.env" ]]; then
        source "$HOME/.rcc/.env"
    fi

    if [[ -z "$RCC_AGENT_TOKEN" ]]; then
        warn "--propose: RCC_AGENT_TOKEN not set. Skipping VibeEngine PROPOSE."
    else
        info "Triggering VibeEngine PROPOSE for hash: $CONTENT_HASH"
        PROPOSE_PAYLOAD="{\"targets\":[\"sparky\"],\"mode\":\"shell\",\"code\":\"echo PROPOSE $CONTENT_HASH | nc -q1 localhost 8080 || true\"}"
        PROPOSE_RESP="$(curl -sf \
            --connect-timeout 5 \
            -H "Authorization: Bearer $RCC_AGENT_TOKEN" \
            -H "Content-Type: application/json" \
            -X POST "$RCC_URL/api/exec" \
            -d "$PROPOSE_PAYLOAD" 2>&1)" || true
        verbose "RCC exec response: $PROPOSE_RESP"
        ok "VibeEngine PROPOSE sent (check RCC exec log for result)"
    fi
fi

# ── Step 5: Optional nano package registry publish ───────────────────────────
# Set --registry <url> to also publish to the nano package registry.
# Requires nano.manifest.json in the source directory (or auto-generates from
# @agentos.name/@agentos.version annotations parsed earlier in this script).
#
# Usage: nanoc publish my_agent.nano --registry http://localhost:7891
#
NANO_REGISTRY="${NANO_REGISTRY:-}"
REGISTRY_TOKEN="${NANO_REGISTRY_TOKEN:-}"

# Parse --registry flag (search remaining argv)
for arg in "${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}"; do
    if [[ "$arg" == --registry=* ]]; then
        NANO_REGISTRY="${arg#--registry=}"
    fi
done

if [[ -n "$NANO_REGISTRY" && -n "$NANO_PKG_NAME" && -n "$NANO_PKG_VERSION" ]]; then
    info "Publishing to nano registry: $NANO_REGISTRY"

    # Build manifest JSON
    MANIFEST_JSON="{\"name\":\"$NANO_PKG_NAME\",\"version\":\"$NANO_PKG_VERSION\",\"main\":\"$(basename "$OUTPUT_WASM")\"}"

    # Sign the tarball with Ed25519 if signing.key exists
    SIG_ARGS=()
    SIGNING_KEY="$HOME/.nanoc/signing.key"
    if [[ -f "$SIGNING_KEY" ]] && command -v openssl &>/dev/null; then
        TARBALL_HASH="$(sha256sum "$OUTPUT_WASM" | awk '{print $1}')"
        TARBALL_HASH_BIN="$(echo -n "$TARBALL_HASH" | xxd -r -p 2>/dev/null || echo "$TARBALL_HASH")"
        # Ed25519 sign: openssl pkeyutl -sign -inkey signing.key -rawin -in <(echo -n "$TARBALL_HASH")
        SIG_B64="$(echo -n "$TARBALL_HASH" | openssl pkeyutl -sign -inkey "$SIGNING_KEY" -rawin 2>/dev/null | base64 -w0 2>/dev/null)" || SIG_B64=""
        PUBKEY_B64="$(openssl pkey -in "$SIGNING_KEY" -pubout -outform DER 2>/dev/null | base64 -w0 2>/dev/null)" || PUBKEY_B64=""
        if [[ -n "$SIG_B64" ]]; then
            SIG_ARGS=(-F "signature=$SIG_B64" -F "pubkey=$PUBKEY_B64")
            verbose "Ed25519 signature: ${SIG_B64:0:20}..."
        fi
    fi

    AUTH_HEADER=()
    [[ -n "$REGISTRY_TOKEN" ]] && AUTH_HEADER=(-H "Authorization: Bearer $REGISTRY_TOKEN")

    PUBLISH_RESP="$(curl -sf \
        --connect-timeout 10 \
        "${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"}" \
        -F "manifest=$MANIFEST_JSON;type=application/json" \
        -F "tarball=@$OUTPUT_WASM;type=application/wasm" \
        "${SIG_ARGS[@]+"${SIG_ARGS[@]}"}" \
        "$NANO_REGISTRY/api/v1/publish" 2>&1)" || { warn "Registry publish failed: $PUBLISH_RESP"; }

    if echo "$PUBLISH_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if d.get('ok') else 1)" 2>/dev/null; then
        ok "Published to registry: ${NANO_PKG_NAME}@${NANO_PKG_VERSION}"
        REGISTRY_SHA="$(echo "$PUBLISH_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('sha256',''))" 2>/dev/null)"
        verbose "  Registry SHA-256: $REGISTRY_SHA"
    else
        warn "Registry publish response: $PUBLISH_RESP"
    fi
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
[[ $CLEANUP_WASM -eq 1 ]] && rm -f "$OUTPUT_WASM"

exit 0
