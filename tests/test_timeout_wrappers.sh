#!/usr/bin/env bash
set -eu

check_wrapper() {
    name=$1

    set +e
    run_wrapper "$name" "$missing_command" >/dev/null 2>&1
    status=$?
    set -e
    if [ "$status" -eq 0 ]; then
        printf '%s permits a missing command to succeed\n' "$name" >&2
        exit 1
    fi

    set +e
    run_wrapper "$name" "$non_executable" >/dev/null 2>&1
    status=$?
    set -e
    if [ "$status" -eq 0 ]; then
        printf '%s permits a non-executable command to succeed\n' "$name" >&2
        exit 1
    fi

    run_wrapper "$name" sh -c 'exit 0'
    set +e
    run_wrapper "$name" sh -c 'exit 23'
    status=$?
    set -e
    if [ "$status" -ne 23 ]; then
        printf '%s changed command exit status 23 to %s\n' "$name" "$status" >&2
        exit 1
    fi
}

run_wrapper() {
    name=$1
    shift
    case "$name" in
        TIMEOUT_CMD) timeout=600 ;;
        BOOTSTRAP2_TIMEOUT_CMD) timeout=3600 ;;
        EXAMPLES_TIMEOUT_CMD|RELEASE_TIMEOUT_CMD) timeout=2400 ;;
        *) return 64 ;;
    esac
    perl -e 'alarm shift @ARGV; exec @ARGV; exit 127' "$timeout" "$@"
}

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/nanolang-timeout-wrapper.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM
missing_command="$tmpdir/missing"
non_executable="$tmpdir/non-executable"
printf '#!/bin/sh\nexit 0\n' > "$non_executable"
chmod 0644 "$non_executable"

check_wrapper TIMEOUT_CMD
check_wrapper BOOTSTRAP2_TIMEOUT_CMD
check_wrapper EXAMPLES_TIMEOUT_CMD
check_wrapper RELEASE_TIMEOUT_CMD
