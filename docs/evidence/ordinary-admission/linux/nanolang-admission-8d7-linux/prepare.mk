include Makefile.gnu
.PHONY: admission-prepare
admission-prepare: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	@printf '%s\n' '$(filter-out $(OBJ_DIR)/nanovm/vm.o,$(NANOVM_OBJECTS)) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /tmp/nanolang-admission-8d7-linux/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /tmp/nanolang-admission-8d7-linux/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /tmp/nanolang-admission-8d7-linux/ldflags.txt
