include Makefile.gnu
.PHONY: map-print-closure
map-print-closure:
	@echo 'CC=$(CC)'
	@echo 'CFLAGS=$(CFLAGS)'
	@echo 'LDFLAGS=$(LDFLAGS)'
	@echo 'OBJECTS=$(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nanovirt/main.o $(SERVICE_DRIVER_OBJECTS) $(FILE_PUBLIC_LIBRARY) lib/libnano_socket_runtime.a lib/libnano_services_runtime.a lib/libnano_websocket_runtime.a'
.PHONY: map-print-consumers
map-print-consumers:
	@echo 'nanoisa=$(NANOISA_DUMP_OBJECT) $(NANOISA_OBJECTS) $(NANOISA_UTF8)'
	@echo 'nvm2c=$(NVM2C_MAIN_OBJECT) $(NANOISA_OBJECTS) $(NANOISA_UTF8) $(FILE_CLI_OBJECT) $(FILE_PUBLIC_LIBRARY) lib/libnano_socket_runtime.a lib/libnano_services_runtime.a lib/libnano_websocket_runtime.a'
	@echo 'nano_vm=$(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nanovm/vmd_protocol.o $(OBJ_DIR)/nanovm/vmd_client.o $(OBJ_DIR)/nanovm/main.o $(FILE_CLI_OBJECT) $(FILE_PUBLIC_LIBRARY) lib/libnano_socket_runtime.a lib/libnano_services_runtime.a lib/libnano_websocket_runtime.a'
	@echo 'export=$(EXPORT_DYNAMIC_LDFLAGS)'
