include Makefile.gnu
.PHONY: portable-prepare
portable-prepare: $(NANOISA_OBJECTS) $(NANOISA_UTF8)
	@printf '%s\n' '$(NANOISA_OBJECTS) $(NANOISA_UTF8)' > /tmp/nanolang-portable-read-corrected-linux/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /tmp/nanolang-portable-read-corrected-linux/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /tmp/nanolang-portable-read-corrected-linux/ldflags.txt
