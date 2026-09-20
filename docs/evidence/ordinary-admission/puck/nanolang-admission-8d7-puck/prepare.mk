include Makefile.gnu
.PHONY: admission-prepare
admission-prepare: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	@printf '%s\n' '$(filter-out $(OBJ_DIR)/nanovm/vm.o,$(NANOVM_OBJECTS)) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /private/tmp/nanolang-admission-8d7-puck/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /private/tmp/nanolang-admission-8d7-puck/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /private/tmp/nanolang-admission-8d7-puck/ldflags.txt
