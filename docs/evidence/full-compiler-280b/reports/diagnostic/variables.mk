include Makefile.gnu
.PHONY: diagnostic-variables
diagnostic-variables:
	@printf "%s\n" "$(COMPILER_OBJECTS)" "$(CFLAGS)" "$(LDFLAGS)"
