include Makefile.gnu
.PHONY: admission-prepare
admission-prepare: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	@printf '%s\n' '$(filter-out $(OBJ_DIR)/nanovm/vm.o,$(NANOVM_OBJECTS)) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /tmp/nanolang-admission-edd-linux/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /tmp/nanolang-admission-edd-linux/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /tmp/nanolang-admission-edd-linux/ldflags.txt
