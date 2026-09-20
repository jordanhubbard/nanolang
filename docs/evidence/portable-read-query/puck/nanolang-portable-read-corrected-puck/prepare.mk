include Makefile.gnu
.PHONY: portable-prepare
portable-prepare: $(NANOISA_OBJECTS) $(NANOISA_UTF8)
	@printf '%s\n' '$(NANOISA_OBJECTS) $(NANOISA_UTF8)' > /private/tmp/nanolang-portable-read-corrected-puck/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /private/tmp/nanolang-portable-read-corrected-puck/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /private/tmp/nanolang-portable-read-corrected-puck/ldflags.txt
