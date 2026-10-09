.PHONY: profile-candidate
profile-candidate:
	$(CC) $(CFLAGS) -iquote src/nanovm -c /private/tmp/nanolang-bootstrap-profile-20261009/vm-profile.c -o /private/tmp/nanolang-bootstrap-profile-20261009/vm-profile.o
	$(CC) $(CFLAGS) -o /private/tmp/nanolang-bootstrap-profile-20261009/nano_vm $(filter-out obj/nanovm/vm.o,$(NANOVM_OBJECTS)) /private/tmp/nanolang-bootstrap-profile-20261009/vm-profile.o $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) obj/nanovm/vmd_protocol.o obj/nanovm/vmd_client.o obj/nanovm/main.o $(FILE_CLI_OBJECT) $(FILE_PUBLIC_LIBRARY) $(LDFLAGS) $(EXPORT_DYNAMIC_LDFLAGS)
