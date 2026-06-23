#!/usr/bin/env bash
# tests/test_dap_smoke.sh — smoke-test the nanolang DAP server
#
# Sends a minimal DAP session over stdin/stdout:
#   initialize → setBreakpoints → configurationDone → (hit bp) → continue → disconnect
#
# Requires: bin/nanolang-dap, python3 (for the driver), a tiny .nano fixture.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DAP_BIN="$REPO_ROOT/bin/nanolang-dap"

if [ ! -x "$DAP_BIN" ]; then
    echo "SKIP: $DAP_BIN not found — run 'make dap' first"
    exit 0
fi

# ---------------------------------------------------------------------------
# Tiny fixture program
# ---------------------------------------------------------------------------
FIXTURE="$(mktemp /tmp/dap_smoke_XXXXXX.nano)"
PYDRIVER="$(mktemp /tmp/dap_driver_XXXXXX.py)"
trap 'rm -f "$FIXTURE" "$PYDRIVER"' EXIT

cat > "$FIXTURE" <<'NANO'
fn main() -> int {
    let x: int = 42
    let y: int = x + 1
    return y
}
NANO

# ---------------------------------------------------------------------------
# Python driver: sends DAP messages and checks responses
# ---------------------------------------------------------------------------
cat > "$PYDRIVER" <<'PYEOF'
import subprocess, sys, json, time

dap_bin  = sys.argv[1]
fixture  = sys.argv[2]
errors   = []
seq      = [0]

proc = subprocess.Popen(
    [dap_bin],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

def send(msg):
    seq[0] += 1
    msg["seq"] = seq[0]
    body = json.dumps(msg).encode()
    header = ("Content-Length: %d\r\n\r\n" % len(body)).encode()
    proc.stdin.write(header + body)
    proc.stdin.flush()

def recv():
    headers = b""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            return None
        headers += ch
        if headers.endswith(b"\r\n\r\n"):
            break
    cl = 0
    for line in headers.decode().splitlines():
        if line.startswith("Content-Length:"):
            cl = int(line.split(":",1)[1].strip())
    body = b""
    while len(body) < cl:
        chunk = proc.stdout.read(cl - len(body))
        if not chunk:
            return None
        body += chunk
    return json.loads(body.decode())

def expect_response(command, success=True):
    for _ in range(20):
        msg = recv()
        if msg is None:
            errors.append("EOF waiting for %s response" % command)
            return None
        if msg.get("type") == "response" and msg.get("command") == command:
            if msg.get("success") != success:
                errors.append("%s: expected success=%s, got %s" % (command, success, msg.get("success")))
            return msg
    errors.append("Did not receive %s response" % command)
    return None

# 1. initialize
send({"type":"request","command":"initialize","arguments":{"clientID":"test","adapterID":"nanolang"}})
r = expect_response("initialize")
assert r and r["success"], "initialize failed"

# Consume initialized event
for _ in range(5):
    msg = recv()
    if msg and msg.get("type") == "event" and msg.get("event") == "initialized":
        break

# 2. launch
send({"type":"request","command":"launch","arguments":{"program": fixture}})
expect_response("launch")

# 3. setBreakpoints (line 3 — the 'let y' line)
send({"type":"request","command":"setBreakpoints","arguments":{
    "source":{"path": fixture},
    "breakpoints":[{"line":3}]
}})
r = expect_response("setBreakpoints")
assert r and r["success"], "setBreakpoints failed"
bps = r.get("body",{}).get("breakpoints",[])
assert len(bps) == 1 and bps[0].get("verified"), "Breakpoint not verified: %s" % bps

# 4. configurationDone -> starts execution
send({"type":"request","command":"configurationDone","arguments":{}})
expect_response("configurationDone")

# 5. Wait for stopped event (breakpoint hit)
stopped = None
for _ in range(30):
    msg = recv()
    if msg is None:
        break
    if msg.get("type") == "event" and msg.get("event") == "stopped":
        stopped = msg
        break
assert stopped is not None, "Did not receive stopped event"
assert stopped["body"].get("reason") in ("breakpoint","step"), \
    "Unexpected stop reason: %s" % stopped["body"]

# 6. threads
send({"type":"request","command":"threads","arguments":{}})
r = expect_response("threads")
assert r and r["body"]["threads"][0]["name"] == "main", "threads wrong"

# 7. stackTrace
send({"type":"request","command":"stackTrace","arguments":{"threadId":1}})
r = expect_response("stackTrace")
assert r and r["body"]["totalFrames"] >= 1, "stackTrace empty"

# 8. scopes
send({"type":"request","command":"scopes","arguments":{"frameId":0}})
r = expect_response("scopes")
scopes = r["body"]["scopes"]
assert any(s["name"] == "Locals" for s in scopes), "No Locals scope"

# 9. variables (locals)
var_ref = next(s["variablesReference"] for s in scopes if s["name"] == "Locals")
send({"type":"request","command":"variables","arguments":{"variablesReference": var_ref}})
r = expect_response("variables")
vars_list = r["body"]["variables"]
names = [v["name"] for v in vars_list]
assert "x" in names, "Expected 'x' in locals, got: %s" % names

# 10. continue
send({"type":"request","command":"continue","arguments":{"threadId":1}})
expect_response("continue")

# 11. Wait for terminated event
for _ in range(30):
    msg = recv()
    if msg is None:
        break
    if msg and msg.get("type") == "event" and msg.get("event") in ("terminated","exited"):
        break

# 12. disconnect
send({"type":"request","command":"disconnect","arguments":{}})
expect_response("disconnect")

proc.stdin.close()
proc.wait(timeout=3)

if errors:
    print("FAIL: " + "; ".join(errors))
    sys.exit(1)
else:
    print("PASS: DAP smoke test")
    sys.exit(0)
PYEOF

python3 "$PYDRIVER" "$DAP_BIN" "$FIXTURE"
