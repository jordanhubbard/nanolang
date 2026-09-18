# GNU make prefers GNUmakefile over Makefile.
# Keep GNU-specific logic in Makefile.gnu and include it here.
include Makefile.gnu

# I qualify finite variant payloads separately from compiler bootstrap.
.PHONY: test-native-variant-array-carriers
test-native-variant-array-carriers: nvm2c nanoisa_dump nano_vm test-nvm2c-shapes
	python3 -m unittest tests.test_native_variant_array_carriers tests.test_native_variant_scalar_carriers

.PHONY: test-native-generic-scalar-array
test-native-generic-scalar-array: nvm2c nano_vm
	python3 -m unittest tests.test_native_generic_scalar_array
