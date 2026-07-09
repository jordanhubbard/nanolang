#!/usr/bin/env bash
# test_registry.sh — integration tests for nano-registry-server.mjs
# Spins up a temporary registry, publishes, searches, installs, and verifies lockfile.
#
# Usage: bash tests/test_registry.sh
# Exit:  0 on pass, 1 on any failure

set -euo pipefail
cd "$(dirname "$0")/.."

PASS=0; FAIL=0
ok()   { echo "  PASS: $1"; ((PASS++)); }
fail() { echo "  FAIL: $1"; ((FAIL++)); }
die()  { echo "error: $*" >&2; exit 1; }

command -v node    >/dev/null || die "node required"
command -v curl    >/dev/null || die "curl required"
command -v python3 >/dev/null || die "python3 required"
command -v tar     >/dev/null || die "tar required"

# ─── Start registry server on ephemeral port ─────────────────────────────────

REG_PORT=39002
STORE_DIR=$(mktemp -d)
TOKEN="test-tok-1234"

export PORT=$REG_PORT STORAGE_DIR="$STORE_DIR" REGISTRY_TOKEN="$TOKEN"
node tools/nano-registry-server.mjs &
REG_PID=$!

cleanup() { kill "$REG_PID" 2>/dev/null || true; rm -rf "$STORE_DIR"; }
trap cleanup EXIT

sleep 0.4

BASE="http://localhost:$REG_PORT"

# ─── /health ─────────────────────────────────────────────────────────────────

echo "--- health check"
resp=$(curl -sf "$BASE/health")
echo "$resp" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); assert d.get('ok') == True" \
    && ok "/health returns ok" || fail "/health"

# ─── /packages (empty) ───────────────────────────────────────────────────────

echo "--- empty list"
resp=$(curl -sf "$BASE/packages")
count=$(echo "$resp" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['count'])")
[[ "$count" == "0" ]] && ok "empty registry count=0" || fail "empty count=$count"

# ─── publish a package ───────────────────────────────────────────────────────

echo "--- publish"
PKG_TMP=$(mktemp -d)
mkdir -p "$PKG_TMP/mylib-0.1.0"
echo 'fn hello() -> String = "hello from mylib"' > "$PKG_TMP/mylib-0.1.0/lib.nano"
tar -czf "$PKG_TMP/mylib-0.1.0.tar.gz" -C "$PKG_TMP" mylib-0.1.0

TARB64=$(base64 < "$PKG_TMP/mylib-0.1.0.tar.gz" | tr -d '\n')
PAYLOAD=$(python3 -c "
import json, sys
m = {'name':'mylib','version':'0.1.0','description':'A test library','author':'test','license':'MIT','tags':['test'],'deps':{},'main':'lib.nano'}
print(json.dumps({'manifest':m,'tarball_b64':sys.argv[1]}))
" "$TARB64")

resp=$(curl -sf -X POST "$BASE/packages" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$PAYLOAD")

ok_flag=$(echo "$resp" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('yes' if d.get('ok') else 'no')")
[[ "$ok_flag" == "yes" ]] && ok "publish mylib@0.1.0" || fail "publish: $resp"
rm -rf "$PKG_TMP"

# ─── publish v0.2.0 ──────────────────────────────────────────────────────────

PKG_TMP=$(mktemp -d)
mkdir -p "$PKG_TMP/mylib-0.2.0"
echo 'fn hello() -> String = "hello v2"' > "$PKG_TMP/mylib-0.2.0/lib.nano"
tar -czf "$PKG_TMP/mylib-0.2.0.tar.gz" -C "$PKG_TMP" mylib-0.2.0
TARB64=$(base64 < "$PKG_TMP/mylib-0.2.0.tar.gz" | tr -d '\n')
PAYLOAD=$(python3 -c "
import json, sys
m = {'name':'mylib','version':'0.2.0','description':'A test library v2','author':'test','license':'MIT','tags':['test','v2'],'deps':{},'main':'lib.nano'}
print(json.dumps({'manifest':m,'tarball_b64':sys.argv[1]}))
" "$TARB64")
curl -sf -X POST "$BASE/packages" -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" -d "$PAYLOAD" > /dev/null
ok "publish mylib@0.2.0"
rm -rf "$PKG_TMP"

# ─── search ──────────────────────────────────────────────────────────────────

echo "--- search"
resp=$(curl -sf "$BASE/search?q=test")
count=$(echo "$resp" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['count'])")
[[ "$count" == "1" ]] && ok "search 'test' returns 1 result" || fail "search count=$count"

resp2=$(curl -sf "$BASE/search?q=nonexistent_xyz")
count2=$(echo "$resp2" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['count'])")
[[ "$count2" == "0" ]] && ok "search 'nonexistent_xyz' returns 0" || fail "search miss=$count2"

# ─── package info + latest ────────────────────────────────────────────────────

echo "--- package info"
resp=$(curl -sf "$BASE/packages/mylib")
latest=$(echo "$resp" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['latest'])")
[[ "$latest" == "0.2.0" ]] && ok "latest updated to 0.2.0" || fail "latest=$latest"

vers_count=$(echo "$resp" | python3 -c "import json,sys; print(len(json.loads(sys.stdin.read())['versions']))")
[[ "$vers_count" == "2" ]] && ok "two versions listed" || fail "vers_count=$vers_count"

# ─── version resolution ──────────────────────────────────────────────────────

echo "--- semver resolution"
resp=$(curl -sf "$BASE/packages/mylib/^0.1.0")
resolved=$(echo "$resp" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('resolved_version',''))")
# ^0.1.0 should resolve to latest 0.x — 0.2.0
[[ "$resolved" == "0.2.0" ]] && ok "^0.1.0 resolves to 0.2.0" || fail "^0.1.0 resolved=$resolved"

resp2=$(curl -sf "$BASE/packages/mylib/0.1.0")
resolved2=$(echo "$resp2" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('resolved_version',''))")
[[ "$resolved2" == "0.1.0" ]] && ok "exact 0.1.0 resolves correctly" || fail "exact resolved=$resolved2"

# ─── tarball download ────────────────────────────────────────────────────────

echo "--- tarball download"
TMP_DOWN=$(mktemp)
curl -sf "$BASE/packages/mylib/0.2.0/tarball" -o "$TMP_DOWN"
sz=$(wc -c < "$TMP_DOWN")
[[ "$sz" -gt 100 ]] && ok "tarball download non-empty ($sz bytes)" || fail "tarball size=$sz"
rm -f "$TMP_DOWN"

# ─── auth rejection ──────────────────────────────────────────────────────────

echo "--- auth"
http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/packages" \
    -H "Content-Type: application/json" -d '{"manifest":{"name":"x","version":"1.0.0"},"tarball_b64":""}')
[[ "$http_code" == "401" ]] && ok "publish without token returns 401" || fail "auth http_code=$http_code"

# ─── 404 for unknown package ──────────────────────────────────────────────────

echo "--- 404"
http_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/packages/nosuchpkg")
[[ "$http_code" == "404" ]] && ok "unknown package returns 404" || fail "404 check=$http_code"

# ─── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ "$FAIL" == "0" ]]
