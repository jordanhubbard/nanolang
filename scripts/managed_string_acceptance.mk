# I prepare the exact original acceptance providers without running suites.
# Invoke from the checkout root with GNU make:
# make -f GNUmakefile -f scripts/managed_string_acceptance.mk managed-string-acceptance-providers
# Compiler/LLVM selections belong to the retained qualification configuration.

.PHONY: managed-string-acceptance-providers
managed-string-acceptance-providers: nvm2llvm nvm2wasm nanoisa_dump nano_vm nvm2c \
	$(OBJ_DIR)/binary64_parser_vm $(OBJ_DIR)/scalar_global_lifetime \
	$(OBJ_DIR)/literal_string_aliases $(OBJ_DIR)/test_verifier_profiles \
	$(OBJ_DIR)/generic_numeric_bits

# I retain the link inputs and flags of the original neighbor Make targets.
$(OBJ_DIR)/scalar_global_lifetime: tests/nanoisa/scalar_global_lifetime.c $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $< $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)

$(OBJ_DIR)/literal_string_aliases: tests/nanoisa/literal_string_aliases.c $(OBJ_DIR)/nanoisa/nvm2llvm.o $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJ_DIR)/nanoisa/nvm2llvm.o $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)

$(OBJ_DIR)/test_verifier_profiles: tests/nanoisa/test_verifier_profiles.c $(OBJ_DIR)/nanoisa/nvm2llvm.o $(NANOISA_OBJECTS) $(NANOISA_UTF8)
	$(CC) $(CFLAGS) -I$(NANOISA_DIR) -o $@ $< $(OBJ_DIR)/nanoisa/nvm2llvm.o $(NANOISA_OBJECTS) $(NANOISA_UTF8) $(LDFLAGS)

$(OBJ_DIR)/generic_numeric_bits: tests/nanoisa/generic_numeric_bits.c $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $< $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
