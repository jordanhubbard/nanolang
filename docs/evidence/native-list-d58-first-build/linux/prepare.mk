include Makefile.gnu
.PHONY: list-prepare
list-prepare: $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) nano_virt nano_vm
	@printf '%s\n' '$(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /home/jkh/nanolang-qualification/record-lists-d58-linux-prepare/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /home/jkh/nanolang-qualification/record-lists-d58-linux-prepare/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /home/jkh/nanolang-qualification/record-lists-d58-linux-prepare/ldflags.txt
