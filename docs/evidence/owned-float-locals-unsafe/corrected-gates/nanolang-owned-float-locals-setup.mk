.PHONY: transitive-wrapper-setup
transitive-wrapper-setup: nanoisa_emit nano_virt nano_vm nvm2c nanoisa_dump test-local-binding-metadata
	$(CC) $(CFLAGS) -o obj/borrow_shadow_names tests/nanovirt/borrow_shadow_names.c $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
