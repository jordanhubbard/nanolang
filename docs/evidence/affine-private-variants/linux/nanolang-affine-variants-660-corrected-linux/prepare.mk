include Makefile.gnu
.PHONY: admission-prepare
admission-prepare: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	@printf '%s\n' '$(filter-out $(OBJ_DIR)/nanovm/vm.o,$(NANOVM_OBJECTS)) $(filter-out $(OBJ_DIR)/nanoisa/service_bindings_module.o,$(NANOISA_OBJECTS)) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /tmp/nanolang-affine-variants-660-corrected-linux/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /tmp/nanolang-affine-variants-660-corrected-linux/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /tmp/nanolang-affine-variants-660-corrected-linux/ldflags.txt
	@printf '%s\n' '$(filter-out $(OBJ_DIR)/nanoisa/affine_state.o $(OBJ_DIR)/nanoisa/nvm_v2_layouts.o,$(NANOISA_OBJECTS)) $(NANOISA_UTF8)' > /tmp/nanolang-affine-variants-660-corrected-linux/variant-providers.txt
