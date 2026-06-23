#!/usr/bin/env bash
set -euo pipefail

status=0

for file in "$@"; do
  if [ ! -f "$file" ]; then
    continue
  fi

  externs=()
  while IFS= read -r line; do
    externs+=("$line")
  done < <(sed -nE 's/^[[:space:]]*extern[[:space:]]+fn[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\1/p' "$file")

  check_all=0
  if git diff --name-status HEAD -- "$file" | grep -q '^A'; then
    check_all=1
  fi

  if [ "$check_all" -eq 1 ]; then
    funcs=()
    while IFS= read -r line; do
      funcs+=("$line")
    done < <(sed -nE 's/^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\2/p' "$file")
  else
    funcs=()
    while IFS= read -r line; do
      funcs+=("$line")
    done < <(git diff -U0 HEAD -- "$file" \
      | sed -nE 's/^\+[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\2/p' \
      | sort -u)
  fi

  if [ ${#funcs[@]} -eq 0 ]; then
    funcs=("")
  fi

  for fn_name in "${funcs[@]}"; do
    if [ -z "$fn_name" ]; then
      continue
    fi
    if [ ${#externs[@]} -gt 0 ]; then
      if printf '%s\n' "${externs[@]}" | grep -qx "$fn_name"; then
        continue
      fi
    fi
    if ! grep -Eq "^[[:space:]]*shadow[[:space:]]+$fn_name\\b" "$file"; then
      echo "Missing shadow test: $file -> $fn_name"
      status=1
    fi
  done
done

exit "$status"
