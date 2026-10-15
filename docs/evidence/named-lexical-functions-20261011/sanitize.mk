NAMED_WORK = /private/tmp/nl51-named-functions
NAMED_SANITIZERS = -O1 -fsanitize=address,undefined -fno-sanitize-recover=all
.PHONY: named-sanitize
named-sanitize:
	$(CC) $(CFLAGS) $(NAMED_SANITIZERS) -c src/nanovm/vm.c -o $(NAMED_WORK)/vm-asan.o
	$(CC) $(CFLAGS) $(NAMED_SANITIZERS) -o $(NAMED_WORK)/test-vm-asan tests/nanovm/test_vm.c $(NAMED_WORK)/vm-asan.o $(filter-out obj/nanovm/vm.o,$(NANOVM_OBJECTS)) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=1 $(NAMED_WORK)/test-vm-asan
	$(CC) $(CFLAGS) $(NAMED_SANITIZERS) -c src/nanoisa/nvm2c.c -o $(NAMED_WORK)/nvm2c-asan.o
	$(CC) $(CFLAGS) $(NAMED_SANITIZERS) -o $(NAMED_WORK)/nvm2c-asan $(NVM2C_MAIN_OBJECT) $(NAMED_WORK)/nvm2c-asan.o $(filter-out obj/nanoisa/nvm2c.o,$(NANOISA_OBJECTS)) $(NANOISA_UTF8) $(FILE_CLI_OBJECT) $(FILE_PUBLIC_LIBRARY) lib/libnano_socket_runtime.a lib/libnano_services_runtime.a lib/libnano_websocket_runtime.a $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=1 $(NAMED_WORK)/nvm2c-asan $(NAMED_WORK)/recursive_capture.nvm -o $(NAMED_WORK)/recursive_capture-asan.c
