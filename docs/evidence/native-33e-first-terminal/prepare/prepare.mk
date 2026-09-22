include Makefile.gnu
.PHONY: list-prepare
list-prepare: $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) nano_virt nano_vm
	@printf '%s\n' '$(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /Users/jkh/nanolang-qualification/record-lists-33e-prepare/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /Users/jkh/nanolang-qualification/record-lists-33e-prepare/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /Users/jkh/nanolang-qualification/record-lists-33e-prepare/ldflags.txt
