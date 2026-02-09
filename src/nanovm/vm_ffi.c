/*
 * NanoVM FFI Bridge - Call native C functions from the VM
 *
 * Module loading and symbol resolution are delegated to the shared
 * ffi_loader. This file handles VM-specific marshaling between
 * NanoValue and C function signatures, plus module introspection.
 */

#include "vm_ffi.h"
#include "runtime/dyn_array.h"
#include "runtime/ffi_loader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>

/* ========================================================================
 * Module Registry (delegates to ffi_loader)
 * ======================================================================== */

static Environment *ffi_env = NULL;  /* For module introspection */

void vm_ffi_set_env(Environment *env) {
    ffi_env = env;
}

void vm_ffi_init(void) {
    ffi_loader_init(false);
}

void vm_ffi_shutdown(void) {
    ffi_loader_shutdown();
}

bool vm_ffi_load_module(const char *module_name) {
    if (!ffi_loader_is_initialized()) ffi_loader_init(false);
    if (ffi_loader_find(module_name)) return true;

    /* Find library using the shared search logic (no module_dir for VM) */
    char path[1024];
    if (!ffi_loader_find_library(module_name, NULL, path, sizeof(path))) {
        /* Not fatal - function might be in main executable or already-loaded lib */
        return false;
    }

    return ffi_loader_open(module_name, path);
}

/* ========================================================================
 * NanoValue ↔ C Marshaling
 * ======================================================================== */

/* Marshal NanoValue args to C void* array for polymorphic dispatch */
static int marshal_args(NanoValue *args, int arg_count,
                        const NvmImportEntry *imp, const uint8_t *param_types,
                        void **arg_ptrs) {
    for (int i = 0; i < arg_count; i++) {
        uint8_t expected_tag = (i < imp->param_count && param_types)
                               ? param_types[i] : args[i].tag;

        switch (expected_tag) {
            case TAG_INT:
                arg_ptrs[i] = (void *)(intptr_t)args[i].as.i64;
                break;
            case TAG_FLOAT:
                /* Float args: store in stack buffer and pass pointer.
                 * NOTE: This is a known limitation - proper float passing
                 * requires libffi. For now, cast to intptr_t which works
                 * on arm64 for many C functions that take double. */
                arg_ptrs[i] = (void *)(intptr_t)args[i].as.i64;
                break;
            case TAG_BOOL:
                arg_ptrs[i] = (void *)(intptr_t)(args[i].as.boolean ? 1 : 0);
                break;
            case TAG_STRING:
                if (args[i].tag == TAG_STRING && args[i].as.string) {
                    arg_ptrs[i] = (void *)vmstring_cstr(args[i].as.string);
                } else {
                    arg_ptrs[i] = (void *)"";
                }
                break;
            case TAG_OPAQUE:
                /* Opaque values stored as raw pointer in i64 */
                arg_ptrs[i] = (void *)(intptr_t)args[i].as.i64;
                break;
            case TAG_ARRAY: {
                /* Convert VmArray to DynArray for C functions */
                VmArray *va = args[i].as.array;
                if (!va) {
                    arg_ptrs[i] = (void *)dyn_array_new(ELEM_INT);
                } else {
                    ElementType et = ELEM_INT;
                    if (va->elem_type == TAG_STRING) et = ELEM_STRING;
                    else if (va->elem_type == TAG_FLOAT) et = ELEM_FLOAT;
                    else if (va->elem_type == TAG_BOOL) et = ELEM_BOOL;
                    DynArray *da = dyn_array_new(et);
                    for (uint32_t j = 0; j < va->length; j++) {
                        NanoValue elem = va->elements[j];
                        switch (et) {
                            case ELEM_INT:   dyn_array_push_int(da, elem.as.i64); break;
                            case ELEM_FLOAT: dyn_array_push_float(da, elem.as.f64); break;
                            case ELEM_BOOL:  dyn_array_push_bool(da, elem.as.boolean); break;
                            case ELEM_STRING:
                                if (elem.as.string)
                                    dyn_array_push_string(da, vmstring_cstr(elem.as.string));
                                else
                                    dyn_array_push_string(da, "");
                                break;
                            default:
                                dyn_array_push_int(da, elem.as.i64);
                                break;
                        }
                    }
                    arg_ptrs[i] = (void *)da;
                }
                break;
            }
            default:
                /* Pass as raw int64 (best effort) */
                arg_ptrs[i] = (void *)(intptr_t)args[i].as.i64;
                break;
        }
    }
    return arg_count;
}

/* Convert C int64_t result to NanoValue based on return type tag */
static NanoValue marshal_result(int64_t raw_result, uint8_t return_tag,
                                VmHeap *heap) {
    switch (return_tag) {
        case TAG_INT:
            return val_int(raw_result);
        case TAG_FLOAT: {
            /* Result is actually a double bit-pattern in int64_t */
            double d;
            memcpy(&d, &raw_result, sizeof(double));
            return val_float(d);
        }
        case TAG_BOOL:
            return val_bool(raw_result != 0);
        case TAG_STRING: {
            const char *str = (const char *)(intptr_t)raw_result;
            if (str) {
                VmString *vs = vm_string_new(heap, str, (uint32_t)strlen(str));
                return val_string(vs);
            }
            return val_void();
        }
        case TAG_VOID:
            return val_void();
        case TAG_OPAQUE: {
            /* Opaque pointer stored as int64 */
            NanoValue v = {0};
            v.tag = TAG_OPAQUE;
            v.as.i64 = raw_result;
            return v;
        }
        case TAG_ARRAY: {
            /* C functions return DynArray* - convert to VmArray */
            DynArray *darr = (DynArray *)(intptr_t)raw_result;
            if (!darr) return val_void();
            VmArray *varr = vm_array_new(heap, TAG_INT, (uint32_t)darr->length);
            for (int64_t ai = 0; ai < darr->length; ai++) {
                NanoValue elem;
                switch (darr->elem_type) {
                    case ELEM_INT:
                        elem = val_int(dyn_array_get_int(darr, ai));
                        break;
                    case ELEM_FLOAT:
                        elem = val_float(dyn_array_get_float(darr, ai));
                        break;
                    case ELEM_BOOL:
                        elem = val_bool(dyn_array_get_bool(darr, ai));
                        break;
                    case ELEM_STRING: {
                        const char *s = dyn_array_get_string(darr, ai);
                        if (s) {
                            VmString *vs = vm_string_new(heap, s, (uint32_t)strlen(s));
                            elem = val_string(vs);
                        } else {
                            elem = val_void();
                        }
                        break;
                    }
                    default:
                        elem = val_int(dyn_array_get_int(darr, ai));
                        break;
                }
                vm_array_push(varr, elem);
            }
            NanoValue v = {0};
            v.tag = TAG_ARRAY;
            v.as.array = varr;
            return v;
        }
        default:
            return val_int(raw_result);
    }
}

/* ========================================================================
 * FFI Call Dispatch
 * ======================================================================== */

typedef int64_t (*FFI_Fn0)(void);
typedef int64_t (*FFI_Fn1)(void *);
typedef int64_t (*FFI_Fn2)(void *, void *);
typedef int64_t (*FFI_Fn3)(void *, void *, void *);
typedef int64_t (*FFI_Fn4)(void *, void *, void *, void *);
typedef int64_t (*FFI_Fn5)(void *, void *, void *, void *, void *);

/* Float-specific dispatch: on arm64, doubles use FP registers, not GP.
 * Using void-ptr and int64_t types puts args in wrong registers for C math fns. */
typedef double (*FFI_DFn0)(void);
typedef double (*FFI_DFn1)(double);
typedef double (*FFI_DFn2)(double, double);
typedef double (*FFI_DFn3)(double, double, double);

/* Check if all params and return are float */
static bool is_all_float_signature(const NvmImportEntry *imp,
                                   const uint8_t *param_types, int arg_count) {
    if (imp->return_type != TAG_FLOAT) return false;
    for (int i = 0; i < arg_count; i++) {
        uint8_t tag = (i < imp->param_count && param_types) ? param_types[i] : TAG_FLOAT;
        if (tag != TAG_FLOAT) return false;
    }
    return true;
}

bool vm_ffi_call(const NvmModule *module, uint32_t import_idx,
                 NanoValue *args, int arg_count,
                 NanoValue *result, VmHeap *heap,
                 char *error_msg, size_t error_msg_size) {
    if (!ffi_loader_is_initialized()) vm_ffi_init();

    if (import_idx >= module->import_count) {
        snprintf(error_msg, error_msg_size, "Import index %u out of range", import_idx);
        return false;
    }

    const NvmImportEntry *imp = &module->imports[import_idx];
    const char *func_name = nvm_get_string(module, imp->function_name_idx);
    const char *mod_name = nvm_get_string(module, imp->module_name_idx);

    if (!func_name) {
        snprintf(error_msg, error_msg_size, "NULL function name for import %u", import_idx);
        return false;
    }

    /* Handle module introspection functions (___module_*) */
    if (func_name && strncmp(func_name, "___module_", 10) == 0 && ffi_env) {
        const char *rest = func_name + 10;

        if (strncmp(rest, "is_unsafe_", 10) == 0) {
            const char *mname = rest + 10;
            ModuleInfo *mi = env_get_module(ffi_env, mname);
            *result = val_bool(mi ? mi->is_unsafe : false);
            return true;
        }
        if (strncmp(rest, "has_ffi_", 8) == 0) {
            const char *mname = rest + 8;
            ModuleInfo *mi = env_get_module(ffi_env, mname);
            *result = val_bool(mi ? mi->has_ffi : false);
            return true;
        }
        if (strncmp(rest, "name_", 5) == 0) {
            const char *mname = rest + 5;
            VmString *vs = vm_string_new(heap, mname, (uint32_t)strlen(mname));
            *result = val_string(vs);
            return true;
        }
        if (strncmp(rest, "path_", 5) == 0) {
            const char *mname = rest + 5;
            ModuleInfo *mi = env_get_module(ffi_env, mname);
            const char *path = (mi && mi->path) ? mi->path : "";
            VmString *vs = vm_string_new(heap, path, (uint32_t)strlen(path));
            *result = val_string(vs);
            return true;
        }
        if (strncmp(rest, "function_count_", 15) == 0) {
            const char *mname = rest + 15;
            ModuleInfo *mi = env_get_module(ffi_env, mname);
            *result = val_int(mi ? mi->function_count : 0);
            return true;
        }
        if (strncmp(rest, "function_name_", 14) == 0) {
            const char *mname = rest + 14;
            ModuleInfo *mi = env_get_module(ffi_env, mname);
            int64_t idx = (arg_count >= 1) ? args[0].as.i64 : 0;
            const char *fn = "";
            if (mi && mi->exported_functions && idx >= 0 && idx < mi->function_count) {
                fn = mi->exported_functions[idx] ? mi->exported_functions[idx] : "";
            }
            VmString *vs = vm_string_new(heap, fn, (uint32_t)strlen(fn));
            *result = val_string(vs);
            return true;
        }
        if (strncmp(rest, "struct_count_", 13) == 0) {
            const char *mname = rest + 13;
            ModuleInfo *mi = env_get_module(ffi_env, mname);
            *result = val_int(mi ? mi->struct_count : 0);
            return true;
        }
        if (strncmp(rest, "struct_name_", 12) == 0) {
            const char *mname = rest + 12;
            ModuleInfo *mi = env_get_module(ffi_env, mname);
            int64_t idx = (arg_count >= 1) ? args[0].as.i64 : 0;
            const char *sn = "";
            if (mi && mi->exported_structs && idx >= 0 && idx < mi->struct_count) {
                sn = mi->exported_structs[idx] ? mi->exported_structs[idx] : "";
            }
            VmString *vs = vm_string_new(heap, sn, (uint32_t)strlen(sn));
            *result = val_string(vs);
            return true;
        }
    }

    /* Try to load the module if we have a module name */
    if (mod_name && mod_name[0] != '\0') {
        vm_ffi_load_module(mod_name);
    }

    /* Resolve the function through the shared loader */
    void *func_ptr = ffi_loader_resolve(func_name);
    if (!func_ptr) {
        snprintf(error_msg, error_msg_size,
                 "FFI: function '%s' not found (module '%s')",
                 func_name, mod_name ? mod_name : "");
        return false;
    }

    /* Marshal arguments */
    void *arg_ptrs[16] = {0};
    if (arg_count > 16) {
        snprintf(error_msg, error_msg_size, "Too many FFI arguments (%d > 16)", arg_count);
        return false;
    }

    const uint8_t *param_types = module->import_param_types
                                 ? module->import_param_types[import_idx] : NULL;

    /* Fast path: all-float signatures use properly typed dispatch so
     * doubles go through FP registers (critical on arm64). */
    if (is_all_float_signature(imp, param_types, arg_count)) {
        double dargs[16];
        for (int i = 0; i < arg_count; i++) {
            dargs[i] = (args[i].tag == TAG_FLOAT) ? args[i].as.f64
                      : (args[i].tag == TAG_INT)   ? (double)args[i].as.i64
                      : 0.0;
        }
        double dresult = 0.0;
        switch (arg_count) {
            case 0: dresult = ((FFI_DFn0)func_ptr)(); break;
            case 1: dresult = ((FFI_DFn1)func_ptr)(dargs[0]); break;
            case 2: dresult = ((FFI_DFn2)func_ptr)(dargs[0], dargs[1]); break;
            case 3: dresult = ((FFI_DFn3)func_ptr)(dargs[0], dargs[1], dargs[2]); break;
            default:
                snprintf(error_msg, error_msg_size,
                         "FFI: unsupported float arg count %d (max 3)", arg_count);
                return false;
        }
        *result = val_float(dresult);
        return true;
    }

    marshal_args(args, arg_count, imp, param_types, arg_ptrs);

    /* Call the function */
    int64_t raw_result = 0;
    switch (arg_count) {
        case 0: raw_result = ((FFI_Fn0)func_ptr)(); break;
        case 1: raw_result = ((FFI_Fn1)func_ptr)(arg_ptrs[0]); break;
        case 2: raw_result = ((FFI_Fn2)func_ptr)(arg_ptrs[0], arg_ptrs[1]); break;
        case 3: raw_result = ((FFI_Fn3)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2]); break;
        case 4: raw_result = ((FFI_Fn4)func_ptr)(arg_ptrs[0], arg_ptrs[1],
                                                  arg_ptrs[2], arg_ptrs[3]); break;
        case 5: raw_result = ((FFI_Fn5)func_ptr)(arg_ptrs[0], arg_ptrs[1],
                                                  arg_ptrs[2], arg_ptrs[3], arg_ptrs[4]); break;
        default:
            snprintf(error_msg, error_msg_size,
                     "FFI: unsupported arg count %d (max 5)", arg_count);
            return false;
    }

    /* Marshal result */
    *result = marshal_result(raw_result, imp->return_type, heap);
    return true;
}

/* ========================================================================
 * Co-Process FFI Isolation
 * ======================================================================== */

#include "cop_protocol.h"
#include <signal.h>
#include <sys/wait.h>

bool vm_ffi_cop_start(VmState *vm, const NvmModule *module) {
    if (vm->cop_pid > 0) return true;  /* Already running */

    /* Serialize the module to send to co-process */
    uint32_t blob_size = 0;
    uint8_t *blob = nvm_serialize(module, &blob_size);
    if (!blob) return false;

    /* Create pipes: parent writes to child stdin, reads from child stdout */
    int pipe_to_child[2];    /* parent writes [1], child reads [0] */
    int pipe_from_child[2];  /* child writes [1], parent reads [0] */

    if (pipe(pipe_to_child) != 0 || pipe(pipe_from_child) != 0) {
        free(blob);
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        free(blob);
        close(pipe_to_child[0]); close(pipe_to_child[1]);
        close(pipe_from_child[0]); close(pipe_from_child[1]);
        return false;
    }

    if (pid == 0) {
        /* Child: set up stdin/stdout from pipes, exec nano_cop */
        close(pipe_to_child[1]);
        close(pipe_from_child[0]);
        dup2(pipe_to_child[0], STDIN_FILENO);
        dup2(pipe_from_child[1], STDOUT_FILENO);
        close(pipe_to_child[0]);
        close(pipe_from_child[1]);

        execlp("nano_cop", "nano_cop", (char *)NULL);
        execl("bin/nano_cop", "nano_cop", (char *)NULL);
        _exit(127);
    }

    /* Parent */
    close(pipe_to_child[0]);
    close(pipe_from_child[1]);
    vm->cop_pid = pid;
    vm->cop_in_fd = pipe_to_child[1];
    vm->cop_out_fd = pipe_from_child[0];

    /* Send INIT with serialized module */
    if (!cop_send(vm->cop_in_fd, COP_MSG_INIT, blob, blob_size)) {
        free(blob);
        vm_ffi_cop_stop(vm);
        return false;
    }
    free(blob);

    /* Wait for READY */
    CopMsgHeader hdr;
    if (!cop_recv_header(vm->cop_out_fd, &hdr) || hdr.msg_type != COP_MSG_READY) {
        vm_ffi_cop_stop(vm);
        return false;
    }

    return true;
}

void vm_ffi_cop_stop(VmState *vm) {
    if (vm->cop_pid <= 0) return;

    /* Try graceful shutdown */
    if (vm->cop_in_fd >= 0) {
        cop_send_simple(vm->cop_in_fd, COP_MSG_SHUTDOWN);
        close(vm->cop_in_fd);
        vm->cop_in_fd = -1;
    }
    if (vm->cop_out_fd >= 0) {
        close(vm->cop_out_fd);
        vm->cop_out_fd = -1;
    }

    /* Wait for child with timeout */
    int status;
    pid_t w = waitpid(vm->cop_pid, &status, WNOHANG);
    if (w == 0) {
        usleep(50000); /* 50ms */
        w = waitpid(vm->cop_pid, &status, WNOHANG);
        if (w == 0) {
            kill(vm->cop_pid, SIGTERM);
            waitpid(vm->cop_pid, &status, 0);
        }
    }

    vm->cop_pid = -1;
}

/* Check if the co-process is still alive. Reaps zombie if dead. */
static bool cop_is_alive(VmState *vm) {
    if (vm->cop_pid <= 0) return false;
    int status;
    pid_t w = waitpid(vm->cop_pid, &status, WNOHANG);
    if (w > 0) {
        /* Child exited (crash or normal exit) — reap it */
        vm->cop_pid = -1;
        if (vm->cop_in_fd >= 0) { close(vm->cop_in_fd); vm->cop_in_fd = -1; }
        if (vm->cop_out_fd >= 0) { close(vm->cop_out_fd); vm->cop_out_fd = -1; }
        return false;
    }
    return true;  /* Still running (w == 0) or error (w < 0, treat as alive) */
}

/* Ensure the co-process is running. Launches on demand or relaunches
 * after a crash. Blocks until the cop is initialized and ready. */
static bool cop_ensure(VmState *vm, const NvmModule *module,
                       char *error_msg, size_t error_msg_size) {
    /* Already running? */
    if (cop_is_alive(vm)) return true;

    /* Launch (or relaunch after crash) */
    if (!vm_ffi_cop_start(vm, module)) {
        snprintf(error_msg, error_msg_size,
                 "COP: failed to launch co-process");
        return false;
    }
    return true;
}

bool vm_ffi_call_cop(VmState *vm, const NvmModule *module, uint32_t import_idx,
                     NanoValue *args, int arg_count,
                     NanoValue *result, VmHeap *heap,
                     char *error_msg, size_t error_msg_size) {
    /* Lazy launch: start cop on first FFI call, or relaunch after crash */
    if (!cop_ensure(vm, module, error_msg, error_msg_size)) {
        /* Could not start cop — fall back to in-process FFI */
        return vm_ffi_call(module, import_idx, args, arg_count,
                           result, heap, error_msg, error_msg_size);
    }

    /* Build request payload: u32 import_idx + u16 argc + serialized args */
    uint8_t payload[8192];
    uint32_t pos = 0;
    memcpy(payload + pos, &import_idx, 4);
    pos += 4;
    uint16_t argc = (uint16_t)arg_count;
    memcpy(payload + pos, &argc, 2);
    pos += 2;

    for (int i = 0; i < arg_count && i < 16; i++) {
        uint32_t n = cop_serialize_value(&args[i], payload + pos, sizeof(payload) - pos);
        if (n == 0) {
            snprintf(error_msg, error_msg_size, "COP: failed to serialize arg %d", i);
            return false;
        }
        pos += n;
    }

    /* Send request */
    if (!cop_send(vm->cop_in_fd, COP_MSG_FFI_REQ, payload, pos)) {
        /* Pipe broken — cop crashed during our call */
        vm_ffi_cop_stop(vm);
        snprintf(error_msg, error_msg_size,
                 "COP: co-process died during FFI request (will relaunch on next call)");
        return false;
    }

    /* Receive response */
    CopMsgHeader hdr;
    if (!cop_recv_header(vm->cop_out_fd, &hdr)) {
        /* Pipe broken — cop crashed while we waited for response */
        vm_ffi_cop_stop(vm);
        snprintf(error_msg, error_msg_size,
                 "COP: co-process died during FFI response (will relaunch on next call)");
        return false;
    }

    if (hdr.msg_type == COP_MSG_FFI_RESULT) {
        if (hdr.payload_len > 0) {
            /* Use stack buffer for small payloads, heap for large (arrays) */
            uint8_t *recv_buf = (hdr.payload_len <= sizeof(payload))
                                ? payload
                                : malloc(hdr.payload_len);
            if (!recv_buf) {
                snprintf(error_msg, error_msg_size, "COP: OOM for result (%u bytes)",
                         hdr.payload_len);
                return false;
            }
            bool ok = cop_recv_payload(vm->cop_out_fd, recv_buf, hdr.payload_len);
            if (!ok) {
                if (recv_buf != payload) free(recv_buf);
                snprintf(error_msg, error_msg_size, "COP: failed to receive result payload");
                return false;
            }
            uint32_t consumed = cop_deserialize_value(recv_buf, hdr.payload_len, result, heap);
            if (recv_buf != payload) free(recv_buf);
            if (consumed == 0) {
                snprintf(error_msg, error_msg_size, "COP: failed to deserialize result");
                return false;
            }
        } else {
            *result = val_void();
        }
        return true;
    } else if (hdr.msg_type == COP_MSG_FFI_ERROR) {
        uint32_t err_len = hdr.payload_len < (uint32_t)(error_msg_size - 1)
                           ? hdr.payload_len : (uint32_t)(error_msg_size - 1);
        if (err_len > 0) {
            cop_recv_payload(vm->cop_out_fd, error_msg, err_len);
            error_msg[err_len] = '\0';
        }
        return false;
    } else {
        snprintf(error_msg, error_msg_size, "COP: unexpected response type 0x%02x",
                 hdr.msg_type);
        return false;
    }
}
