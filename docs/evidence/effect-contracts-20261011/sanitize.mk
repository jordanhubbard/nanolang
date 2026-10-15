EFFECT_WORK = /private/tmp/nl51-effect-contracts
EFFECT_SAN = -O1 -fno-inline -fsanitize=address,undefined -fno-sanitize-recover=all
.PHONY: effect-contract-sanitize
effect-contract-sanitize:
	$(CC) $(CFLAGS) $(EFFECT_SAN) -c src/parser.c -o $(EFFECT_WORK)/parser-asan.o
	$(CC) $(CFLAGS) $(EFFECT_SAN) -c src/typechecker.c -o $(EFFECT_WORK)/typechecker-asan.o
	$(CC) $(CFLAGS) $(EFFECT_SAN) -c src/nominal_types.c -o $(EFFECT_WORK)/nominal-types-asan.o
	$(CC) $(CFLAGS) $(EFFECT_SAN) -o $(EFFECT_WORK)/test-parser-asan tests/test_parser.c $(EFFECT_WORK)/parser-asan.o $(EFFECT_WORK)/typechecker-asan.o $(EFFECT_WORK)/nominal-types-asan.o $(filter-out obj/parser.o obj/typechecker.o obj/nominal_types.o,$(COMMON_OBJECTS)) $(RUNTIME_OBJECTS) $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=1 $(EFFECT_WORK)/test-parser-asan

.PHONY: effect-results-sanitize
effect-results-sanitize:
	$(CC) $(CFLAGS) $(EFFECT_SAN) -o $(EFFECT_WORK)/test-effect-results-asan $(EFFECT_WORK)/effect-results-only.c $(EFFECT_WORK)/parser-asan.o $(EFFECT_WORK)/typechecker-asan.o $(EFFECT_WORK)/nominal-types-asan.o $(filter-out obj/parser.o obj/typechecker.o obj/nominal_types.o,$(COMMON_OBJECTS)) $(RUNTIME_OBJECTS) $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=1 ASAN_SYMBOLIZER_PATH=/opt/homebrew/opt/llvm/bin/llvm-symbolizer $(EFFECT_WORK)/test-effect-results-asan

.PHONY: effect-parser-baseline
effect-parser-baseline:
	$(CC) $(CFLAGS) $(EFFECT_SAN) -c $(EFFECT_WORK)/parser-before.c -o $(EFFECT_WORK)/parser-before-asan.o
	$(CC) $(CFLAGS) -Itests $(EFFECT_SAN) -o $(EFFECT_WORK)/test-parser-before-asan $(EFFECT_WORK)/test-parser-before.c $(EFFECT_WORK)/parser-before-asan.o $(EFFECT_WORK)/typechecker-asan.o $(EFFECT_WORK)/nominal-types-asan.o $(filter-out obj/parser.o obj/typechecker.o obj/nominal_types.o,$(COMMON_OBJECTS)) $(RUNTIME_OBJECTS) $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=1 ASAN_SYMBOLIZER_PATH=/opt/homebrew/opt/llvm/bin/llvm-symbolizer $(EFFECT_WORK)/test-parser-before-asan
