include Makefile.gnu
.PHONY: list-prepare
list-prepare: $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) nano_virt nano_vm
	@printf '%s\n' '$(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /tmp/nanolang-record-lists-1b8-linux-prepare/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /tmp/nanolang-record-lists-1b8-linux-prepare/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /tmp/nanolang-record-lists-1b8-linux-prepare/ldflags.txt
