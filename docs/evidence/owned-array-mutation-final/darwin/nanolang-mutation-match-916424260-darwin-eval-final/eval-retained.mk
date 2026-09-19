eval-retained: stage1 $(OBJ_DIR)/test_interpreter_ffi_native.so $(OBJ_DIR)/eval_io_faults.o $(OBJ_DIR)/eval_clock_test.o
	$(CC) $(CFLAGS) -o /private/tmp/nanolang-mutation-match-916424260-darwin-eval-final/test_eval tests/test_eval.c $(filter-out $(OBJ_DIR)/eval.o $(OBJ_DIR)/eval/eval_io.o,$(COMMON_OBJECTS)) $(OBJ_DIR)/eval_clock_test.o $(OBJ_DIR)/eval_io_faults.o $(RUNTIME_OBJECTS) $(LDFLAGS)
