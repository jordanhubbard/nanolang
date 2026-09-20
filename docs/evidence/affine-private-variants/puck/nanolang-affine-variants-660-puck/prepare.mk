include Makefile.gnu
.PHONY: admission-prepare
admission-prepare: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	@printf '%s\n' '$(filter-out $(OBJ_DIR)/nanovm/vm.o,$(NANOVM_OBJECTS)) $(filter-out $(OBJ_DIR)/nanoisa/service_bindings_module.o,$(NANOISA_OBJECTS)) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /private/tmp/nanolang-affine-variants-660-puck/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /private/tmp/nanolang-affine-variants-660-puck/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /private/tmp/nanolang-affine-variants-660-puck/ldflags.txt
	@printf '%s\n' '$(filter-out $(OBJ_DIR)/nanoisa/affine_state.o $(OBJ_DIR)/nanoisa/nvm_v2_layouts.o,$(NANOISA_OBJECTS)) $(NANOISA_UTF8)' > /private/tmp/nanolang-affine-variants-660-puck/variant-providers.txt
