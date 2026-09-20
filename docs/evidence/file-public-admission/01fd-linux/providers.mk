
.PHONY: public-providers public-config
public-providers: $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o $(OBJ_DIR)/nanovirt/wrapper_gen.o $(OBJ_DIR)/nanovm/vmd_protocol.o $(OBJ_DIR)/nanovm/vmd_client.o
public-config:
	@printf '%s\n' 'ALL=$(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o' 'ISA=$(NANOISA_OBJECTS) $(NANOISA_UTF8)' 'OPCODES=$(OBJ_DIR)/nanovirt/wrapper_gen.o $(OBJ_DIR)/nanoisa/nvm2llvm.o $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o' 'LDFLAGS=$(LDFLAGS)'
