#!/usr/bin/env bash
set -euo pipefail

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "$source_dir" rev-parse --show-toplevel)"
obj_dir="${OBJ_DIR:-$repo_dir/_build}"
check_only=0
if [[ "${1:-}" == "--check" ]]; then
  check_only=1
fi

python_bin=""
for candidate in \
  "$obj_dir/doc-toolchain/bin/python3" \
  "$obj_dir/doc-toolchain/bin/python" \
  "$obj_dir/doc-toolchain/Scripts/python.exe"
do
  if [[ -x "$candidate" ]]; then
    python_bin="$candidate"
    break
  fi
done
if [[ -z "$python_bin" && -z "${OBJ_DIR:-}" && -x "$repo_dir/.venv/bin/python3" ]]; then
  python_bin="$repo_dir/.venv/bin/python3"
fi
if [[ -z "$python_bin" ]]; then
  python_bin="python3"
fi

if ! "$python_bin" -c "import pptx, docx, PIL, lxml" >/dev/null 2>&1; then
  echo "Pinned authoring toolchain is not importable for ${python_bin}." >&2
  echo "Run \`make doc-toolchain-bootstrap\` to install python-pptx, python-docx, lxml, and Pillow into ignored OBJ_DIR." >&2
  echo "Refusing to pip-install from regenerate_python.sh without that explicit Makefile authorization." >&2
  exit 2
fi

if [[ "$check_only" -eq 1 ]]; then
  printf '%s\n' "$python_bin"
  exit 0
fi

build_dir="$obj_dir/nanolang-developer-overview"
mkdir -p "$build_dir"

(
  cd "$source_dir"
  NANOLANG_DECK_SOURCE="$source_dir" \
  NANOLANG_DECK_REPO="$repo_dir" \
  OBJ_DIR="$obj_dir" \
  "$python_bin" build_deck.py
  NANOLANG_DECK_SOURCE="$source_dir" \
  NANOLANG_DECK_REPO="$repo_dir" \
  OBJ_DIR="$obj_dir" \
  "$python_bin" build_narrative.py
  if [[ "${NANOLANG_DECK_SKIP_RENDER:-}" != "1" ]]; then
    NANOLANG_DECK_SOURCE="$source_dir" \
    NANOLANG_DECK_REPO="$repo_dir" \
    OBJ_DIR="$obj_dir" \
    "$python_bin" render_slides.py
  fi
)

pptx="${NANOLANG_DECK_OUTPUT:-$source_dir/nanolang-developer-overview.pptx}"
docx="${NANOLANG_NARRATIVE_OUTPUT:-$source_dir/nanolang-developer-overview.docx}"
manifest="$build_dir/capability-manifest.json"
echo "PPTX: $pptx"
echo "DOCX: $docx"

acceptance="$build_dir/acceptance.json"
"$python_bin" - "$manifest" "$acceptance" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest.get("slides") == 12
assert Path(manifest["narrative"]).is_file()
Path(sys.argv[2]).write_text(json.dumps({"slides": 12, "accepted": True}, indent=2) + "\n")
print("NanoLang document pair accepted")
PY
echo "Acceptance report: $acceptance"
