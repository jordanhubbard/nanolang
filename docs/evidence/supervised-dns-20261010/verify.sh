#!/bin/sh
set -eu
cd /Users/jordanh/Src/nanolang
for variant in clang gcc sanitizer; do
 case "$variant" in
 clang) compiler=/opt/homebrew/opt/llvm/bin/clang; flags='' ;;
 gcc) compiler=/opt/homebrew/bin/gcc-16; flags='' ;;
 sanitizer) compiler=/opt/homebrew/opt/llvm/bin/clang; flags='-fsanitize=address,undefined -fno-sanitize-recover=all' ;;
 esac
 "$compiler" -std=c11 -D_DEFAULT_SOURCE -D_DARWIN_C_SOURCE -g -O1 -Wall -Wextra -Werror $flags src/nsi_socket_resolver_main.c src/nsi_socket.c src/nsi_cap.c -o "/private/tmp/nl51-dns-helper-$variant"
 "$compiler" -std=c11 -D_DEFAULT_SOURCE -D_DARWIN_C_SOURCE -g -O1 -Wall -Wextra -Werror $flags tests/test_nsi_socket_resolver.c src/nsi_socket_resolver.c src/nsi_socket.c src/nsi_cap.c -o "/private/tmp/nl51-dns-test-$variant"
 ASAN_OPTIONS=detect_leaks=1 "/private/tmp/nl51-dns-test-$variant" "/private/tmp/nl51-dns-helper-$variant"
done
