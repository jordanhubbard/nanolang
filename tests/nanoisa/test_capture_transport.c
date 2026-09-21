/* I qualify transport and refusal, never execution of capture-bearing code. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "assembler.h"
#include "capture_bindings.h"
#include "disassembler.h"
#include "isa.h"
#include "nvm2c.h"
#include "nvm_v2_sections.h"
#include "verifier.h"
#ifdef CAPTURE_TRANSPORT_CONSUMERS
#include "nvm2llvm.h"
#include "vm.h"
#include "portable_host_plan.h"
#include "owned_array_admission.h"
#include "mixed_samples_internal.h"
/* I provide the process arguments required by linked runtime providers. */
int g_argc = 0;
char **g_argv = NULL;
#endif

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
    fprintf(stderr, "I failed line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

#ifdef CAPTURE_TRANSPORT_ALLOCATION_TEST
/* I interpose only the capture payload malloc and decoder table calloc sites.
 * Other module allocations retain their real allocator and existing coverage. */
static void *owned[8];
static size_t owned_count, attempts, fail_at, failures;
static bool armed, persistent;
static bool refuse_allocation(void) {
    if (!armed) return false;
    size_t index = attempts++;
    bool refuse = index == fail_at || (persistent && index > fail_at);
    if (refuse) ++failures;
    return refuse;
}
static void remember(void *pointer) {
    if (!armed || !pointer) return;
    CHECK(owned_count < sizeof owned / sizeof owned[0]);
    owned[owned_count++] = pointer;
}
void *capture_test_malloc(size_t size) {
    if (refuse_allocation()) return NULL;
    void *pointer = malloc(size);
    remember(pointer);
    return pointer;
}
void *capture_test_calloc(size_t count, size_t size) {
    if (refuse_allocation()) return NULL;
    void *pointer = calloc(count, size);
    remember(pointer);
    return pointer;
}
void capture_test_free(void *pointer) {
    for (size_t i = 0; i < owned_count; ++i) {
        if (owned[i] == pointer) {
            owned[i] = owned[--owned_count];
            break;
        }
    }
    free(pointer);
}
#endif

static NvmModule *fixture(bool sites, bool captures) {
    static const uint8_t ordinary[] = {OP_RET, OP_RET, OP_RET, OP_RET};
    static const uint8_t closures[] = {
        OP_CLOSURE_BIND,0,0,0,0, OP_CLOSURE_BIND,1,0,0,0,
        OP_CLOSURE_BIND,2,0,0,0, OP_RET,
        OP_CLOSURE_BIND,3,0,0,0, OP_RET, OP_RET, OP_RET
    };
    static const uint8_t local0[] = {NVM_CAPTURE_SHARED, NVM_CAPTURE_VALUE};
    static const uint8_t local1[] = {NVM_CAPTURE_VALUE};
    static const uint8_t upvalues1[] = {
        NVM_CAPTURE_SHARED, NVM_CAPTURE_VALUE, NVM_CAPTURE_SHARED
    };
    static const uint8_t upvalues2[] = {NVM_CAPTURE_SHARED};
    static const uint8_t repeated[] = {
        NVM_CAPTURE_LOCAL,NVM_CAPTURE_SHARED,0,0,
        NVM_CAPTURE_LOCAL,NVM_CAPTURE_VALUE,1,0,
        NVM_CAPTURE_LOCAL,NVM_CAPTURE_SHARED,0,0
    };
    static const uint8_t forwarded[] = {NVM_CAPTURE_UPVALUE,NVM_CAPTURE_SHARED,0,0};
    const char *names[] = {"main", "nested", "forwarded", "empty"};
    const uint32_t offsets[] = {0,16,22,23}, lengths[] = {16,6,1,1};
    NvmModule *m = nvm_module_new();
    CHECK(m != NULL);
    nvm_append_code(m, sites ? closures : ordinary,
                    sites ? sizeof closures : sizeof ordinary);
    CHECK(m->code_size == (sites ? sizeof closures : sizeof ordinary));
    NvmCaptureFunction functions[4] = {
        {2,0,local0,NULL}, {1,sites ? 3 : 0,local1,sites ? upvalues1 : NULL},
        {0,sites ? 1 : 0,NULL,sites ? upvalues2 : NULL}, {0,0,NULL,NULL}
    };
    for (uint32_t i = 0; i < 4; ++i) {
        NvmFunctionEntry fn = {0};
        fn.name_idx = nvm_add_string(m, names[i], (uint32_t)strlen(names[i]));
        CHECK(fn.name_idx != UINT32_MAX);
        fn.local_count = functions[i].local_count;
        fn.upvalue_count = functions[i].upvalue_count;
        fn.code_offset = sites ? offsets[i] : i;
        fn.code_length = sites ? lengths[i] : 1;
        fn.result_tag = TAG_VOID;
        CHECK(nvm_add_function(m, &fn) == i);
    }
    m->header.flags |= NVM_FLAG_HAS_MAIN;
    m->header.entry_point = 0;
    if (captures) {
        NvmCaptureSite rows[] = {
            {0,0,1,3,repeated}, {0,5,1,3,repeated},
            {0,10,3,0,NULL}, {1,0,2,1,forwarded}
        };
        NvmCaptureBindings binding = {functions,sites ? rows : NULL,4,sites ? 4 : 0,0};
        size_t size = 0;
        CHECK(nvm_capture_bindings_encode(&binding, m, NVM_CAPTURE_TRANSPORT_BYTES,
                                          &m->capture_data, &size) == NVM_CAPTURE_OK);
        CHECK(size > 0 && size <= UINT32_MAX);
        m->capture_size = (uint32_t)size;
        CHECK(nvm_capture_bindings_validate_module(m) == NVM_CAPTURE_OK);
    }
    return m;
}

static uint8_t *wire(const NvmModule *m, size_t *size) {
    NvmV2Module view = {0};
    CHECK(nvm_v2_from_nvm_module(m, &view) == NVM_V2_OK);
    CHECK(view.capture_data == m->capture_data && view.capture_size == m->capture_size);
    CHECK(nvm_v2_module_serialize(&view, NULL, 0, size) == NVM_V2_OK);
    uint8_t *bytes = malloc(*size);
    CHECK(bytes != NULL);
    CHECK(nvm_v2_module_serialize(&view, bytes, *size, size) == NVM_V2_OK);
    nvm_v2_module_free(&view);
    CHECK(nvm_capture_bindings_validate_module(m) == NVM_CAPTURE_OK);
    return bytes;
}

static void refusal(const NvmModule *m) {
    CHECK(!nvm_verify(m).ok);
    CHECK(!nvm_verify_function(m, 0).ok);
    CHECK(!nvm_verify_affine_function(m, 0).ok);
    CHECK(!nvm_verify_owned_module(m).ok);
    uint32_t legacy_size = 999;
    CHECK(nvm_serialize(m, &legacy_size) == NULL && legacy_size == 0);
    char error[256] = {0};
    CHECK(nvm2c_emit(m, error, sizeof error) == NULL);
    CHECK(strstr(error, "capture") != NULL);
    NvmModule *plain = fixture(false, false);
    CHECK(nvm_verify(plain).ok);
    const NvmModule *links[] = {m};
    CHECK(!nvm_verify_linked(plain, links, 1).ok);
    nvm_module_free(plain);
}

static void roundtrip(bool sites) {
    NvmModule *source = fixture(sites, true);
    refusal(source);
    size_t size = 0;
    uint8_t *bytes = wire(source, &size);
    uint8_t *saved_wire = malloc(size);
    uint8_t *saved_payload = malloc(source->capture_size);
    CHECK(saved_wire && saved_payload);
    memcpy(saved_wire, bytes, size);
    const uint32_t payload_size = source->capture_size;
    memcpy(saved_payload, source->capture_data, payload_size);
    NvmV2Header header;
    CHECK(nvm_v2_read_header(bytes, size, &header) == NVM_V2_OK);
    CHECK((header.feature_bits & NVM_V2_FEATURE_CAPTURE_BINDINGS) != 0);
    NvmV2Module decoded = {0};
    CHECK(nvm_v2_module_deserialize(bytes, size, &decoded) == NVM_V2_OK);
    CHECK(decoded.capture_size == payload_size);
    CHECK(memcmp(decoded.capture_data, saved_payload, payload_size) == 0);
    NvmModule *copy = NULL;
    CHECK(nvm_v2_to_nvm_module(&decoded, &copy) == NVM_V2_OK && copy);
    CHECK(copy->capture_data != decoded.capture_data);
    CHECK(copy->capture_data != source->capture_data);
    nvm_v2_module_free(&decoded);
    free(bytes);
    nvm_module_free(source);
    /* I exercise the independent owner after both borrowed sources die. */
    CHECK(copy->capture_size == payload_size);
    CHECK(memcmp(copy->capture_data, saved_payload, payload_size) == 0);
    CHECK(nvm_capture_bindings_validate_module(copy) == NVM_CAPTURE_OK);
    refusal(copy);
    size_t copied_size = 0;
    uint8_t *copied_wire = wire(copy, &copied_size);
    CHECK(copied_size == size && memcmp(copied_wire, saved_wire, size) == 0);
    free(copied_wire);
    char *text = disasm_module_styled(copy, DISASM_STYLE_CANONICAL);
    CHECK(text && strstr(text, ".capture_bindings \""));
    AsmResult result;
    NvmModule *assembled = asm_assemble_unverified(text, &result);
    CHECK(assembled && result.error == ASM_OK);
    CHECK(assembled->capture_size == payload_size);
    CHECK(memcmp(assembled->capture_data, saved_payload, payload_size) == 0);
    CHECK(assembled->code_size == copy->code_size);
    CHECK(memcmp(assembled->code, copy->code, copy->code_size) == 0);
    uint8_t *text_wire = wire(assembled, &copied_size);
    CHECK(copied_size == size && memcmp(text_wire, saved_wire, size) == 0);
    free(text_wire);
    CHECK(asm_assemble(text, &result) == NULL && result.error == ASM_ERR_VERIFY);
    free(text);
    nvm_module_free(assembled);
    nvm_module_free(copy);
    free(saved_payload);
    free(saved_wire);
}

#ifdef CAPTURE_TRANSPORT_ALLOCATION_TEST
static size_t allocation_attempt(const NvmModule *source, const NvmV2Module *input,
                                 unsigned operation, size_t failure, bool lasting) {
    NvmModule sentinel = {0}, *copy = &sentinel;
    NvmV2Module view = {0};
    NvmV2Result result = NVM_V2_OK;
    char *text = NULL;
    NvmV2Module before;
    memcpy(&before, input, sizeof before);
    uint8_t payload[256];
    CHECK(source->capture_size <= sizeof payload);
    memcpy(payload, source->capture_data, source->capture_size);
    CHECK(owned_count == 0);
    attempts = failures = 0;
    fail_at = failure;
    persistent = lasting;
    armed = true;
    if (operation == 0) result = nvm_v2_to_nvm_module(input, &copy);
    else if (operation == 1) result = nvm_v2_from_nvm_module(source, &view);
    else text = disasm_module_styled(source, DISASM_STYLE_CANONICAL);
    armed = false;
    size_t count = attempts;
    if (failure != SIZE_MAX) {
        CHECK(failures > 0);
        if (operation < 2) CHECK(result == NVM_V2_ERR_TRUNCATED);
        if (operation == 0) CHECK(copy == NULL);
        if (operation == 2) CHECK(text == NULL);
    } else {
        CHECK(failures == 0 && result == NVM_V2_OK);
        if (operation == 0) {
            CHECK(copy != NULL && copy != &sentinel);
            CHECK(copy->capture_data != source->capture_data);
            CHECK(copy->capture_size == source->capture_size);
            CHECK(memcmp(copy->capture_data, payload, copy->capture_size) == 0);
        }
        if (operation == 1) CHECK(view.capture_data == source->capture_data);
        if (operation == 2) CHECK(text && strstr(text, ".capture_bindings"));
    }
    if (operation == 0 && copy && copy != &sentinel) nvm_module_free(copy);
    nvm_v2_module_free(&view);
    free(text);
    CHECK(owned_count == 0);
    CHECK(memcmp(input, &before, sizeof before) == 0);
    CHECK(memcmp(source->capture_data, payload, source->capture_size) == 0);
    CHECK(nvm_capture_bindings_validate_module(source) == NVM_CAPTURE_OK);
    return count;
}

static void allocation_controls(void) {
    NvmModule *source = fixture(true, true);
    NvmV2Module input = {0};
    CHECK(nvm_v2_from_nvm_module(source, &input) == NVM_V2_OK);
    for (unsigned operation = 0; operation < 3; ++operation) {
        size_t count = allocation_attempt(source, &input, operation, SIZE_MAX, false);
        CHECK(count == (operation == 0 ? 3 : 2));
        for (unsigned lasting = 0; lasting < 2; ++lasting) {
            for (size_t index = 0; index < count; ++index) {
                allocation_attempt(source, &input, operation, index, lasting != 0);
                CHECK(allocation_attempt(source, &input, operation, SIZE_MAX, false) == count);
            }
        }
    }
    nvm_v2_module_free(&input);
    nvm_module_free(source);
    CHECK(owned_count == 0);
}
#endif

#ifdef CAPTURE_TRANSPORT_CONSUMERS
static void private_consumer_controls(const NvmModule *module) {
    /* These aligned markers are never dereferenced: refusal must preserve them. */
    max_align_t marker;
    NvmPortableReadPlan *portable = (NvmPortableReadPlan *)&marker;
    NvmOwnedArrayLayouts *layouts = (NvmOwnedArrayLayouts *)&marker;
    NvmOwnedArrayOrigins *origins = (NvmOwnedArrayOrigins *)&marker;
    NvmOwnedArrayPlan *owner = (NvmOwnedArrayPlan *)&marker;
    NvmMixedSamplesPlan *mixed = (NvmMixedSamplesPlan *)&marker;
    CHECK(module->service_data == NULL && module->service_size == 0);
    CHECK(nvm_capture_bindings_validate_module(module) == NVM_CAPTURE_OK);
    uint8_t *payload = malloc(module->capture_size);
    CHECK(payload != NULL);
    memcpy(payload, module->capture_data, module->capture_size);

    NvmPortableReadResult pr = nvm_portable_read_plan(module, &portable);
    CHECK(pr.status == NVM_PORTABLE_READ_UNSUPPORTED);
    CHECK(strcmp(pr.message, "I require service and capture admission before portable execution.") == 0);
    CHECK(portable == (NvmPortableReadPlan *)&marker);
    NvmRecordPlanResult lr = nvm_describe_owned_array_layouts(module, &layouts);
    CHECK(lr.status == NVM_RECORD_UNRESOLVED);
    CHECK(strcmp(lr.message, "I require service and capture admission before owner-array description.") == 0);
    CHECK(layouts == (NvmOwnedArrayLayouts *)&marker);
    NvmOwnerOriginResult or = nvm_analyze_owned_array_origins(module, &origins);
    CHECK(or.status == NVM_OWNER_ORIGIN_UNRESOLVED);
    CHECK(strcmp(or.message, "I require service and capture admission before origin analysis.") == 0);
    CHECK(origins == (NvmOwnedArrayOrigins *)&marker);
    NvmOwnerAuthorityResult ar = nvm_owned_array_admit(module, &owner);
    CHECK(ar.status == NVM_OWNER_AUTH_UNRESOLVED);
    CHECK(strcmp(ar.message, "I require service and capture execution admission before owner ARRAY admission.") == 0);
    CHECK(owner == (NvmOwnedArrayPlan *)&marker);
    NvmMixedShapeResult mr = nvm_mixed_samples_prepare(module, &mixed);
    CHECK(mr.status == NVM_MIXED_SHAPE_UNRESOLVED);
    CHECK(strcmp(mr.message, "I refuse service contracts before mixed preparation.") == 0);
    CHECK(mixed == (NvmMixedSamplesPlan *)&marker);
    CHECK(nvm_owned_array_route(module) == NVM_OWNER_ARRAY_NOT_SELECTED);
    CHECK(memcmp(payload, module->capture_data, module->capture_size) == 0);
    free(payload);

    NvmModule *plain = fixture(false, false);
    pr = nvm_portable_read_plan(plain, &portable);
    CHECK(pr.status == NVM_PORTABLE_READ_NOT_SELECTED);
    CHECK(portable == (NvmPortableReadPlan *)&marker);
    CHECK(nvm_owned_array_route(plain) == NVM_OWNER_ARRAY_NOT_SELECTED);
    nvm_module_free(plain);
}

static void consumer_controls(void) {
    /* Ordinary RET bodies isolate metadata admission from opcode refusal. */
    NvmModule *module = fixture(false, true);
    private_consumer_controls(module);
    for (unsigned entry = 0; entry < 4; ++entry) {
        VmState *vm = calloc(1, sizeof *vm);
        CHECK(vm != NULL);
        vm_init(vm, module);
        CHECK(!vm->verified);
        NanoValue output = val_int(731);
        VmResult result;
        if (entry == 0) result = vm_execute(vm);
        else if (entry == 1) result = vm_call_function(vm, 0, NULL, 0);
        else if (entry == 2) result = vm_invoke(vm, 0, NULL, 0, &output);
        else result = vm_invoke_callable(vm, val_function(0), NULL, 0, &output);
        CHECK(result == VM_ERR_TYPE_ERROR);
        CHECK(output.tag == TAG_INT && output.as.i64 == 731);
        CHECK(vm->frame_count == 0 && vm->stack_size == 0);
        vm_destroy(vm);
        free(vm);
    }
    NvmModule *plain = fixture(false, false);
    VmState *vm = calloc(1, sizeof *vm);
    CHECK(vm != NULL);
    vm_init(vm, plain);
    CHECK(vm_execute(vm) == VM_OK);
    CHECK(vm_link_module(vm, module) == UINT32_MAX);
    CHECK(vm->linked_module_count == 0);
    vm_destroy(vm);
    free(vm);
    nvm_module_free(plain);
    for (unsigned target = 0; target < 2; ++target) {
        FILE *stream = tmpfile();
        CHECK(stream != NULL);
        static const char sentinel[] = "existing output";
        CHECK(fwrite(sentinel, 1, sizeof sentinel, stream) == sizeof sentinel);
        char error[256] = {0};
        CHECK(!nvm2llvm_emit_target(module, stream, error, sizeof error,
                                   "main", (NvmLlvmTarget)target));
        CHECK(strstr(error, "capture") != NULL);
        CHECK(ftell(stream) == (long)sizeof sentinel);
        rewind(stream);
        char output[sizeof sentinel];
        CHECK(fread(output, 1, sizeof output, stream) == sizeof output);
        CHECK(memcmp(output, sentinel, sizeof output) == 0);
        CHECK(fclose(stream) == 0);
    }
    nvm_module_free(module);
}
#endif

int main(void) {
#ifdef CAPTURE_TRANSPORT_CONSUMERS
    consumer_controls();
#endif
    roundtrip(false);
    roundtrip(true);
#ifdef CAPTURE_TRANSPORT_ALLOCATION_TEST
    allocation_controls();
#endif
    printf("I passed %u capture transport checks.\n", checks);
    return 0;
}
