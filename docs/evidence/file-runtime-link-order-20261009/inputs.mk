include Makefile.gnu
.PHONY: inspect-file-link-order
inspect-file-link-order:
	@printf '%s\n' '$(FILE_RUNTIME_TEST_INPUTS)' > /private/tmp/nanolang-file-link-before.txt
	@printf '%s\n' '$(FILE_RUNTIME_TEST_OBJECTS)' > /private/tmp/nanolang-file-link-after.txt
