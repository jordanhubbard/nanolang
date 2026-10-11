ROW_WORK = /private/tmp/nl51-row-signatures
ROW_SAN = -O1 -fno-inline -fsanitize=address,undefined -fno-sanitize-recover=all
.PHONY: row-sanitize
row-sanitize:
	$(CC) $(CFLAGS) $(ROW_SAN) -c src/parser.c -o $(ROW_WORK)/parser-asan.o
	$(CC) $(CFLAGS) $(ROW_SAN) -c src/env.c -o $(ROW_WORK)/env-asan.o
	$(CC) $(CFLAGS) $(ROW_SAN) -c src/nominal_types.c -o $(ROW_WORK)/nominal-asan.o
	$(CC) $(CFLAGS) $(ROW_SAN) -o $(ROW_WORK)/test-row-asan $(ROW_WORK)/row-only.c $(ROW_WORK)/parser-asan.o $(ROW_WORK)/env-asan.o $(ROW_WORK)/nominal-asan.o $(filter-out obj/parser.o obj/env.o obj/nominal_types.o,$(COMMON_OBJECTS)) $(RUNTIME_OBJECTS) $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=1 ASAN_SYMBOLIZER_PATH=/opt/homebrew/opt/llvm/bin/llvm-symbolizer $(ROW_WORK)/test-row-asan
