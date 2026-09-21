
.PHONY: array-providers array-config
array-providers: $(NANOISA_OBJECTS) $(NANOISA_UTF8)
array-config:
	@printf '%s\n' 'ISA=$(NANOISA_OBJECTS) $(NANOISA_UTF8)' 'LDFLAGS=$(LDFLAGS)'
