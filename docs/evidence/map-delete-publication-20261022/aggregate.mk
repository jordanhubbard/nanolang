# Scoped integration gate only; fresh bootstrap/stage gate remains separate.
include Makefile.gnu
.PHONY: map-delete-publication-scoped
map-delete-publication-scoped: nano_virt nano_vm nanoisa_dump nvm2c
	$(CC) $(CFLAGS) -o $(OBJ_DIR)/borrow_shadow_names tests/nanovirt/borrow_shadow_names.c $(MAP_FIELD_TEST_OBJECTS) $(LDFLAGS)
	$(CC) $(CFLAGS) -o $(OBJ_DIR)/test_map_publication tests/nanovirt/test_map_publication.c $(filter-out $(OBJ_DIR)/nanovirt/codegen.o,$(MAP_FIELD_TEST_OBJECTS)) $(LDFLAGS)
	$(OBJ_DIR)/test_map_publication tests/nanoisa/fixtures/map_record_fields.nano
	MAP_PUBLICATION_COMPILERS=/private/tmp/nanolang-map-delete-publication-20261022/canonical MAP_PUBLICATION_RAW_DRIVERS=/private/tmp/nanolang-map-delete-publication-20261022/raw MAP_PUBLICATION_CC="$(CC)" python3 -m unittest -f -v tests.test_map_field_publication
