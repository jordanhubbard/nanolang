/* I expose actual VM outcomes to the independent Sail stack model. */
#include "nanovm/vm.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc = 0;
char **g_argv = NULL;

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

int main(void) {
    char line[2048];
    while (fgets(line, sizeof(line), stdin)) {
        /* A fixed local prefix also tests that locals are not operands. */
        if ((line[0] != '0' && line[0] != '2') || line[1] != ' ') return 2;
        uint16_t locals = (uint16_t)(line[0] - '0');
        size_t length = strlen(line + 2);
        if (!length || line[length + 1] != '\n') return 2;
        length--;
        if (length % 2 || length / 2 >= 1024) return 2;
        uint8_t code[1024];
        for (size_t i = 0; i < length / 2; i++) {
            int hi = hex_digit(line[2 + 2 * i]);
            int lo = hex_digit(line[3 + 2 * i]);
            if (hi < 0 || lo < 0) return 2;
            code[i] = (uint8_t)((hi << 4) | lo);
        }
        code[length / 2] = OP_HALT;
        NvmModule *module = nvm_module_new();
        if (!module) return 3;
        NvmFunctionEntry fn = {0};
        fn.name_idx = nvm_add_string(module, "main", 4);
        fn.code_offset = nvm_append_code(module, code, (uint32_t)(length / 2 + 1));
        fn.code_length = (uint32_t)(length / 2 + 1);
        fn.local_count = locals;
        module->header.entry_point = nvm_add_function(module, &fn);
        module->header.flags = NVM_FLAG_HAS_MAIN;
        VmState vm;
        vm_init(&vm, module);
        VmResult result = vm_execute(&vm);
        if (result == VM_ERR_STACK_UNDERFLOW) {
            puts("underflow");
        } else if (result == VM_OK && vm.stack_size >= locals) {
            printf("ok");
            for (uint32_t i = vm.stack_size; i > locals; i--) {
                NanoValue v = vm.stack[i - 1];
                if (v.tag != TAG_INT) return 4;
                uint64_t bits;
                memcpy(&bits, &v.as.i64, sizeof(bits));
                printf(" %016" PRIx64, bits);
            }
            putchar('\n');
        } else {
            fprintf(stderr, "I encountered an unexpected VM result: %d.\n", result);
            return 5;
        }
        /* Locals must survive both successful operations and rejected ones. */
        if (vm.stack_size < locals) return 6;
        for (uint16_t i = 0; i < locals; i++)
            if (vm.stack[i].tag != TAG_VOID) return 7;
        vm_destroy(&vm);
        nvm_module_free(module);
    }
    return ferror(stdin) ? 2 : 0;
}
