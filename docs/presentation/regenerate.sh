#!/usr/bin/env bash
set -euo pipefail

# Regenerate the local NanoLang developer deck and narrative through the
# portable python-pptx/python-docx toolchain in ignored OBJ_DIR.

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "$source_dir" rev-parse --show-toplevel)"
obj_dir="${OBJ_DIR:-$repo_dir/_build}"

if OBJ_DIR="$obj_dir" "$source_dir/regenerate_python.sh" --check >/dev/null 2>&1; then
  exec env OBJ_DIR="$obj_dir" "$source_dir/regenerate_python.sh"
fi

discover_codex_artifact_tool_setup() {
  local root="${CODEX_HOME:-$HOME/.codex}"
  local presentations="$root/plugins/cache/openai-primary-runtime/presentations"
  if [[ ! -d "$presentations" ]]; then
    return 1
  fi
  local found
  found="$(find "$presentations" -path '*/container_tools/setup_artifact_tool_workspace.mjs' -print 2>/dev/null | sort | tail -1 || true)"
  if [[ -z "$found" ]]; then
    return 1
  fi
  printf '%s\n' "$found"
}

setup_script=""
if setup_script="$(discover_codex_artifact_tool_setup)"; then
  scratch_dir="$(mktemp -d "${TMPDIR:-/tmp}/nanolang-developer-deck.XXXXXX")"
  node "$setup_script" --workspace "$scratch_dir"
  cp "$source_dir/build_deck.mjs" "$scratch_dir/build_deck.mjs"

  (
    cd "$scratch_dir"
    NANOLANG_DECK_SOURCE="$source_dir" \
    NANOLANG_DECK_REPO="$repo_dir" \
    NANOLANG_DECK_WORKSPACE="$scratch_dir/render" \
    node build_deck.mjs
  )

  echo "Rendered QA files: $scratch_dir/render"
  echo "PPTX: $source_dir/nanolang-developer-overview.pptx"

  python3 "$repo_dir/scripts/verify_document_pair.py" \
    --manifest "$scratch_dir/render/capability-manifest.json" \
    --json "$scratch_dir/render/qa/acceptance.json"
  echo "Acceptance report: $scratch_dir/render/qa/acceptance.json"
  exit 0
fi

echo "Pinned authoring toolchain is not importable, and no Codex presentations plugin was found." >&2
echo "Run \`make doc-toolchain-bootstrap\` to install python-pptx, python-docx, lxml, and Pillow into ignored OBJ_DIR." >&2
echo "The Codex presentations plugin remains an optional fallback only." >&2
exit 2
