include Makefile.gnu
.PHONY: list-prepare
list-prepare: $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)
	@printf '%s\n' '$(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > /Users/jkh/nanolang-qualification/tuple-e456-prepare/providers.txt
	@printf '%s\n' '$(CFLAGS)' > /Users/jkh/nanolang-qualification/tuple-e456-prepare/cflags.txt
	@printf '%s\n' '$(LDFLAGS)' > /Users/jkh/nanolang-qualification/tuple-e456-prepare/ldflags.txt
