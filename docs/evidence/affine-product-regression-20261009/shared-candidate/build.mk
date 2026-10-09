.PHONY: mixed-candidate
mixed-candidate:
	$(CC) $(CFLAGS) -o /private/tmp/nanolang-mixed-record-candidate/nano_vm $(NANOVM_OBJECTS) $(filter-out obj/nanoisa/affine_state.o obj/nanoisa/affine_bytecode.o obj/nanoisa/nvm2c.o,$(NANOISA_OBJECTS)) /private/tmp/nanolang-mixed-record-candidate/c/affine_state.o /private/tmp/nanolang-mixed-record-candidate/c/affine_bytecode.o /private/tmp/nanolang-mixed-record-candidate/c/nvm2c.o $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) obj/nanovm/vmd_protocol.o obj/nanovm/vmd_client.o obj/nanovm/main.o $(FILE_CLI_OBJECT) $(FILE_PUBLIC_LIBRARY) $(LDFLAGS) $(EXPORT_DYNAMIC_LDFLAGS)
	$(CC) $(CFLAGS) -o /private/tmp/nanolang-mixed-record-candidate/nvm2c obj/nanoisa/nvm2c_main.o $(filter-out obj/nanoisa/affine_state.o obj/nanoisa/affine_bytecode.o obj/nanoisa/nvm2c.o,$(NANOISA_OBJECTS)) /private/tmp/nanolang-mixed-record-candidate/c/affine_state.o /private/tmp/nanolang-mixed-record-candidate/c/affine_bytecode.o /private/tmp/nanolang-mixed-record-candidate/c/nvm2c.o $(NANOISA_UTF8) $(FILE_CLI_OBJECT) $(FILE_PUBLIC_LIBRARY) $(LDFLAGS)
.PHONY: mixed-candidate-tests
CANDIDATE_NANOISA = $(filter-out obj/nanoisa/affine_state.o obj/nanoisa/affine_bytecode.o obj/nanoisa/nvm2c.o,$(NANOISA_OBJECTS)) /private/tmp/nanolang-mixed-record-candidate/c/affine_state.o /private/tmp/nanolang-mixed-record-candidate/c/affine_bytecode.o /private/tmp/nanolang-mixed-record-candidate/c/nvm2c.o
mixed-candidate-tests:
	$(CC) $(CFLAGS) -I/private/tmp/nanolang-mixed-record-candidate/c -I$(NANOISA_DIR) -o /private/tmp/nanolang-mixed-record-candidate/test_affine_state tests/nanoisa/test_affine_state.c $(CANDIDATE_NANOISA) $(NANOISA_UTF8) $(LDFLAGS)
	/private/tmp/nanolang-mixed-record-candidate/test_affine_state
	$(CC) $(CFLAGS) -I/private/tmp/nanolang-mixed-record-candidate/c -I$(NANOISA_DIR) -o /private/tmp/nanolang-mixed-record-candidate/test_affine_bytecode tests/nanoisa/test_affine_bytecode.c $(CANDIDATE_NANOISA) $(NANOISA_UTF8) $(LDFLAGS)
	/private/tmp/nanolang-mixed-record-candidate/test_affine_bytecode
	$(CC) $(CFLAGS) -I/private/tmp/nanolang-mixed-record-candidate/c -I$(NANOISA_DIR) -o /private/tmp/nanolang-mixed-record-candidate/test_nvm2c tests/nanoisa/test_nvm2c.c $(CANDIDATE_NANOISA) $(NANOISA_UTF8) $(LDFLAGS)
	/private/tmp/nanolang-mixed-record-candidate/test_nvm2c
.PHONY: mixed-candidate-native
mixed-candidate-native:
	$(CC) $(CFLAGS) -I/private/tmp/nanolang-mixed-record-candidate/c -I$(NANOISA_DIR) -Imodules/nanoisa -o /private/tmp/nanolang-mixed-record-candidate/test_nvm2c tests/nanoisa/test_nvm2c.c $(CANDIDATE_NANOISA) $(NANOISA_UTF8) $(LDFLAGS)
	/private/tmp/nanolang-mixed-record-candidate/test_nvm2c
.PHONY: mixed-candidate-assembler
mixed-candidate-assembler:
	$(CC) $(CFLAGS) -o /private/tmp/nanolang-mixed-record-candidate/nanoisa obj/nanoisa/dump_main.o $(CANDIDATE_NANOISA) $(NANOISA_UTF8) $(FILE_CLI_OBJECT) $(FILE_PUBLIC_LIBRARY) $(LDFLAGS)
.PHONY: mixed-candidate-alloc
mixed-candidate-alloc:
	$(CC) $(CFLAGS) -I$(NANOISA_DIR) -iquote src/nanoisa -Dmalloc=affine_test_malloc -Dcalloc=affine_test_calloc -c /private/tmp/nanolang-mixed-record-candidate/c/affine_state.c -o /private/tmp/nanolang-mixed-record-candidate/c/state_alloc.o
	$(CC) $(CFLAGS) -I/private/tmp/nanolang-mixed-record-candidate/c -I$(NANOISA_DIR) -DAFFINE_ALLOCATION_TEST -o /private/tmp/nanolang-mixed-record-candidate/state_alloc tests/nanoisa/test_affine_state.c /private/tmp/nanolang-mixed-record-candidate/c/state_alloc.o $(filter-out /private/tmp/nanolang-mixed-record-candidate/c/affine_state.o,$(CANDIDATE_NANOISA)) $(NANOISA_UTF8) $(LDFLAGS)
	/private/tmp/nanolang-mixed-record-candidate/state_alloc
	$(CC) $(CFLAGS) -I$(NANOISA_DIR) -iquote src/nanoisa -Dmalloc=affine_bytecode_test_malloc -Dcalloc=affine_bytecode_test_calloc -Drealloc=affine_bytecode_test_realloc -DNVM_AFFINE_TEST_VISIT_LIMIT=affine_bytecode_test_visit_limit -c /private/tmp/nanolang-mixed-record-candidate/c/affine_bytecode.c -o /private/tmp/nanolang-mixed-record-candidate/c/bytecode_alloc.o
	$(CC) $(CFLAGS) -I/private/tmp/nanolang-mixed-record-candidate/c -I$(NANOISA_DIR) -DAFFINE_BYTECODE_ALLOCATION_TEST -o /private/tmp/nanolang-mixed-record-candidate/bytecode_alloc tests/nanoisa/test_affine_bytecode.c /private/tmp/nanolang-mixed-record-candidate/c/bytecode_alloc.o $(filter-out /private/tmp/nanolang-mixed-record-candidate/c/affine_bytecode.o,$(CANDIDATE_NANOISA)) $(NANOISA_UTF8) $(LDFLAGS)
	/private/tmp/nanolang-mixed-record-candidate/bytecode_alloc
