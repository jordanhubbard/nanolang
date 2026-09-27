include Makefile.gnu
.PHONY: list-prepare
list-prepare: $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) nano_virt nano_vm
	@printf '%s\n' '$(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /private/tmp/nanolang-record-lists-1b8-puck-prepare/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /private/tmp/nanolang-record-lists-1b8-puck-prepare/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /private/tmp/nanolang-record-lists-1b8-puck-prepare/ldflags.txt
