include Makefile.gnu
.PHONY: admission-prepare
admission-prepare: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	@printf '%s\n' '$(filter-out $(OBJ_DIR)/nanovm/vm.o,$(NANOVM_OBJECTS)) $(filter-out $(OBJ_DIR)/nanoisa/service_bindings_module.o,$(NANOISA_OBJECTS)) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /tmp/nanolang-service-admission-1f9-linux/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /tmp/nanolang-service-admission-1f9-linux/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /tmp/nanolang-service-admission-1f9-linux/ldflags.txt
