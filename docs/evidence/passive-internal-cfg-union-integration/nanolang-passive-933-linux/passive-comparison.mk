
.PHONY: passive-comparator passive-scoped-sanitizer
passive-comparator: $(NANOISA_OBJECTS) $(NANOISA_UTF8)
	$(CC) $(CFLAGS) -I$(NANOISA_DIR) -I$(NANOISA_MODULE_DIR) -o tests/nanoisa/test_nanoisa_src_nano tests/nanoisa/test_nanoisa_src_nano.c $(NANOISA_OBJECTS) $(NANOISA_UTF8) $(LDFLAGS)
passive-scoped-sanitizer: $(NANOISA_OBJECTS) $(NANOISA_UTF8)
	$(CC) $(CFLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer -DPASSIVE_CFG_ALLOCATION_TEST -I$(NANOISA_DIR) -o obj/test_passive_cfg_sanitized tests/nanoisa/test_passive.c $(filter-out $(OBJ_DIR)/nanoisa/passive.o,$(NANOISA_OBJECTS)) $(NANOISA_UTF8) $(LDFLAGS)
	obj/test_passive_cfg_sanitized
