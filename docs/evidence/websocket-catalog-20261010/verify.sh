#!/bin/sh
set -eu
cd /Users/jordanh/Src/nanolang
for variant in gcc sanitizer; do
 case "$variant" in
 gcc) compiler=/opt/homebrew/bin/gcc-16; flags='' ;;
 sanitizer) compiler=/opt/homebrew/opt/llvm/bin/clang; flags='-fsanitize=address,undefined -fno-sanitize-recover=all' ;;
 esac
 for mode in instrumented linked; do
  if [ "$mode" = instrumented ]; then extra='-DWEBSOCKET_PLAN_INSTRUMENT'; else extra='src/nsi_websocket_plan.c'; fi
  "$compiler" -std=c11 -D_DEFAULT_SOURCE -D_DARWIN_C_SOURCE -g -O1 -Wall -Wextra -Werror $flags tests/test_nsi_websocket_plan.c $extra src/nsi.c src/utf8.c src/cJSON.c -o "/private/tmp/nl51-websocket-plan-$variant-$mode"
  ASAN_OPTIONS=detect_leaks=1 "/private/tmp/nl51-websocket-plan-$variant-$mode" tests/fixtures/nsi_websocket_plan.json
 done
done
