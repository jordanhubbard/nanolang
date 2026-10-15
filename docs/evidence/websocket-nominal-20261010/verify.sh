#!/bin/sh
set -eu
cd /Users/jordanh/Src/nanolang
for variant in gcc sanitizer; do
 case "$variant" in
 gcc) compiler=/opt/homebrew/bin/gcc-16; flags='' ;;
 sanitizer) compiler=/opt/homebrew/opt/llvm/bin/clang; flags='-fsanitize=address,undefined -fno-sanitize-recover=all' ;;
 esac
 for mode in instrumented linked; do
  if [ "$mode" = instrumented ]; then extra='-DNOMINAL_INSTRUMENT'; else extra='src/nanoisa/service_websocket_nominal_plan.c'; fi
  "$compiler" -std=c11 -D_DEFAULT_SOURCE -g -O1 -Wall -Wextra -Werror $flags tests/nanoisa/test_websocket_nominal.c $extra src/nanoisa/service_websocket_nominal.c src/nanoisa/service_file_nominal.c src/nanoisa/service_socket_nominal.c src/nanoisa/nvm_v2_cursor.c src/nsi_websocket_plan.c -o "/private/tmp/nl51-websocket-nominal-$variant-$mode"
  ASAN_OPTIONS=detect_leaks=1 "/private/tmp/nl51-websocket-nominal-$variant-$mode"
 done
done
