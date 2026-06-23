/*
 * NanoVM FFI Bridge - Call native C functions from the VM
 *
 * Module loading and symbol resolution are delegated to the shared
 * ffi_loader. This file handles VM-specific marshaling between
 * NanoValue and C function signatures, plus module introspection.
 */

/* usleep(), kill(), fork(), pipe(), exec*() need _GNU_SOURCE */

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
typedef int64_t (*FFI_Fn6)(void *, void *, void *, void *, void *, void *);
typedef int64_t (*FFI_Fn7)(void *, void *, void *, void *, void *, void *, void *);
typedef int64_t (*FFI_Fn8)(void *, void *, void *, void *, void *, void *, void *, void *);
typedef int64_t (*FFI_Fn9)(void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef int64_t (*FFI_Fn10)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);

/* Float-specific dispatch: on arm64, doubles use FP registers, not GP.
 * Using void-ptr and int64_t types puts args in wrong registers for C math fns. */
typedef double (*FFI_DFn0)(void);
typedef double (*FFI_DFn1)(double);
typedef double (*FFI_DFn2)(double, double);
typedef double (*FFI_DFn3)(double, double, double);
typedef double (*FFI_DFn4)(double, double, double, double);
typedef double (*FFI_DFn5)(double, double, double, double, double);
typedef double (*FFI_DFn6)(double, double, double, double, double, double);
typedef double (*FFI_DFn7)(double, double, double, double, double, double, double);
typedef double (*FFI_DFn8)(double, double, double, double, double, double, double, double);
typedef double (*FFI_DFn9)(double, double, double, double, double, double, double, double, double);
typedef double (*FFI_DFn10)(double, double, double, double, double, double, double, double, double, double);

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
            case 4: dresult = ((FFI_DFn4)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3]); break;
            case 5: dresult = ((FFI_DFn5)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4]); break;
            case 6: dresult = ((FFI_DFn6)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5]); break;
            case 7: dresult = ((FFI_DFn7)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5], dargs[6]); break;
            case 8: dresult = ((FFI_DFn8)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5], dargs[6], dargs[7]); break;
            case 9: dresult = ((FFI_DFn9)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5], dargs[6], dargs[7], dargs[8]); break;
            case 10: dresult = ((FFI_DFn10)func_ptr)(dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5], dargs[6], dargs[7], dargs[8], dargs[9]); break;
            default:
                snprintf(error_msg, error_msg_size,
                         "FFI: unsupported float arg count %d (max 10)", arg_count);
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
        case 6: raw_result = ((FFI_Fn6)func_ptr)(arg_ptrs[0], arg_ptrs[1],
                                                  arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5]); break;
        case 7: raw_result = ((FFI_Fn7)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2],
                                                  arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6]); break;
        case 8: raw_result = ((FFI_Fn8)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2],
                                                  arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7]); break;
        case 9: raw_result = ((FFI_Fn9)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2],
                                                  arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8]); break;
        case 10: raw_result = ((FFI_Fn10)func_ptr)(arg_ptrs[0], arg_ptrs[1], arg_ptrs[2],
                                                   arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8], arg_ptrs[9]); break;
        default:
            snprintf(error_msg, error_msg_size,
                     "FFI: unsupported arg count %d (max 10)", arg_count);
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
#include <sys/mman.h>
#include <sys/wait.h>
#include <poll.h>

/* MAP_ANON is BSD; MAP_ANONYMOUS is the POSIX/Linux name */
#ifndef MAP_ANON
#  ifdef MAP_ANONYMOUS
#    define MAP_ANON MAP_ANONYMOUS
#  else
#    define MAP_ANON 0x1000  /* macOS value */
#  endif
#endif

/* ========================================================================
 * Co-process lifecycle (shared-memory mailbox fast path)
 * ======================================================================== */

bool vm_ffi_cop_start(VmState *vm, const NvmModule *module) {
    if (vm->cop_pid > 0) return true;

    /* Create the shared-memory mailbox — inherited across fork() */
    size_t mbox_size = sizeof(CopMailbox);
    CopMailbox *mailbox = (CopMailbox *)mmap(NULL, mbox_size,
                                             PROT_READ | PROT_WRITE,
                                             MAP_SHARED | MAP_ANON, -1, 0);
    if (mailbox == MAP_FAILED) return false;
    memset(mailbox, 0, mbox_size);

    /* Two 1-byte signal pipes (replace the old full-payload pipes) */
    int sig_to_child[2], sig_from_child[2];
    if (pipe(sig_to_child) != 0 || pipe(sig_from_child) != 0) {
        munmap(mailbox, mbox_size);
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        munmap(mailbox, mbox_size);
        close(sig_to_child[0]);  close(sig_to_child[1]);
        close(sig_from_child[0]); close(sig_from_child[1]);
        return false;
    }

    if (pid == 0) {
        /* Child: close parent-side pipe ends and run the cop logic inline
         * (no exec — mailbox pointer is valid because we forked, not exec'd) */
        close(sig_to_child[1]);
        close(sig_from_child[0]);
        cop_child_main(mailbox, mbox_size,
                       sig_to_child[0], sig_from_child[1], module);
        _exit(0);  /* cop_child_main never returns normally */
    }

    /* Parent: close child-side pipe ends */
    close(sig_to_child[0]);
    close(sig_from_child[1]);

    vm->cop_pid = pid;
    vm->cop_mailbox = mailbox;
    vm->cop_mailbox_size = mbox_size;
    vm->cop_sig_send_fd = sig_to_child[1];
    vm->cop_sig_recv_fd = sig_from_child[0];
    vm->cop_in_fd  = -1;  /* not used in mailbox mode */
    vm->cop_out_fd = -1;

    /* Wait for ready signal from child (up to 5 s) */
    struct pollfd pfd = { .fd = vm->cop_sig_recv_fd, .events = POLLIN };
    if (poll(&pfd, 1, 5000) <= 0) {
        vm_ffi_cop_stop(vm);
        return false;
    }
    uint8_t byte;
    if (read(vm->cop_sig_recv_fd, &byte, 1) != 1) {
        vm_ffi_cop_stop(vm);
        return false;
    }

    return true;
}

void vm_ffi_cop_stop(VmState *vm) {
    if (vm->cop_pid <= 0) return;

    /* Close the send pipe — child sees EOF and exits cleanly */
    if (vm->cop_sig_send_fd >= 0) {
        close(vm->cop_sig_send_fd);
        vm->cop_sig_send_fd = -1;
    }
    if (vm->cop_sig_recv_fd >= 0) {
        close(vm->cop_sig_recv_fd);
        vm->cop_sig_recv_fd = -1;
    }
    /* Legacy pipe fds (may be -1 in mailbox mode) */
    if (vm->cop_in_fd >= 0) {
        cop_send_simple(vm->cop_in_fd, COP_MSG_SHUTDOWN);
        close(vm->cop_in_fd);
        vm->cop_in_fd = -1;
    }
    if (vm->cop_out_fd >= 0) {
        close(vm->cop_out_fd);
        vm->cop_out_fd = -1;
    }

    /* Wait up to 50 ms for graceful exit, then SIGTERM */
    int status;
    pid_t w = waitpid(vm->cop_pid, &status, WNOHANG);
    if (w == 0) {
        usleep(50000);
        w = waitpid(vm->cop_pid, &status, WNOHANG);
        if (w == 0) {
            kill(vm->cop_pid, SIGTERM);
            waitpid(vm->cop_pid, &status, 0);
        }
    }
    vm->cop_pid = -1;

    if (vm->cop_mailbox) {
        munmap(vm->cop_mailbox, vm->cop_mailbox_size);
        vm->cop_mailbox = NULL;
        vm->cop_mailbox_size = 0;
    }
}

/* Check if the co-process is still alive, reaping zombie if dead. */
static bool cop_is_alive(VmState *vm) {
    if (vm->cop_pid <= 0) return false;
    int status;
    pid_t w = waitpid(vm->cop_pid, &status, WNOHANG);
    if (w > 0) {
        vm->cop_pid = -1;
        if (vm->cop_sig_send_fd >= 0) { close(vm->cop_sig_send_fd); vm->cop_sig_send_fd = -1; }
        if (vm->cop_sig_recv_fd >= 0) { close(vm->cop_sig_recv_fd); vm->cop_sig_recv_fd = -1; }
        if (vm->cop_in_fd  >= 0) { close(vm->cop_in_fd);  vm->cop_in_fd  = -1; }
        if (vm->cop_out_fd >= 0) { close(vm->cop_out_fd); vm->cop_out_fd = -1; }
        if (vm->cop_mailbox) {
            munmap(vm->cop_mailbox, vm->cop_mailbox_size);
            vm->cop_mailbox = NULL;
            vm->cop_mailbox_size = 0;
        }
        return false;
    }
    return true;
}

static bool cop_ensure(VmState *vm, const NvmModule *module,
                       char *error_msg, size_t error_msg_size) {
    if (cop_is_alive(vm)) return true;
    if (!vm_ffi_cop_start(vm, module)) {
        snprintf(error_msg, error_msg_size, "COP: failed to launch co-process");
        return false;
    }
    return true;
}

/* ========================================================================
 * vm_ffi_call_cop — fast mailbox path + pipe fallback
 * ======================================================================== */

bool vm_ffi_call_cop(VmState *vm, const NvmModule *module, uint32_t import_idx,
                     NanoValue *args, int arg_count,
                     NanoValue *result, VmHeap *heap,
                     char *error_msg, size_t error_msg_size) {
    if (!cop_ensure(vm, module, error_msg, error_msg_size)) {
        return vm_ffi_call(module, import_idx, args, arg_count,
                           result, heap, error_msg, error_msg_size);
    }

    CopMailbox *mbox = vm->cop_mailbox;

    /* ── Fast path: mailbox ──────────────────────────────────────────── */
    if (mbox) {
        /* Serialize args directly into the mailbox request slot */
        uint32_t pos = 0;
        bool fits = true;
        for (int i = 0; i < arg_count && i < 16 && fits; i++) {
            uint32_t n = cop_serialize_value(&args[i],
                                             mbox->req_data + pos,
                                             COP_MAILBOX_SLOT_SIZE - pos);
            if (n == 0) { fits = false; break; }
            pos += n;
        }

        if (fits) {
            mbox->req_import_idx = import_idx;
            mbox->req_argc       = (uint16_t)arg_count;
            mbox->req_data_size  = (uint16_t)pos;

            /* Wake the child — pipe write is a full memory barrier on POSIX */
            uint8_t sig = 1;
            if (write(vm->cop_sig_send_fd, &sig, 1) != 1) {
                vm_ffi_cop_stop(vm);
                snprintf(error_msg, error_msg_size,
                         "COP: signal pipe broken (child crash?)");
                return false;
            }

            /* Wait for response with per-call timeout */
            struct pollfd pfd = { .fd = vm->cop_sig_recv_fd, .events = POLLIN };
            int n = poll(&pfd, 1, vm->cop_timeout_ms);
            if (n == 0) {
                /* Timeout: kill cop, return error */
                vm_ffi_cop_stop(vm);
                snprintf(error_msg, error_msg_size,
                         "COP: timeout after %d ms", vm->cop_timeout_ms);
                return false;
            }
            if (n < 0 || read(vm->cop_sig_recv_fd, &sig, 1) != 1) {
                vm_ffi_cop_stop(vm);
                snprintf(error_msg, error_msg_size, "COP: ack pipe broken");
                return false;
            }

            /* Read result from mailbox */
            if (mbox->resp_is_error) {
                snprintf(error_msg, error_msg_size, "%s", mbox->resp_error);
                return false;
            }
            if (mbox->resp_data_size > 0) {
                uint32_t consumed = cop_deserialize_value(
                    mbox->resp_data, mbox->resp_data_size, result, heap);
                if (consumed == 0) {
                    snprintf(error_msg, error_msg_size,
                             "COP: failed to deserialize mailbox result");
                    return false;
                }
            } else {
                *result = val_void();
            }
            return true;
        }
        /* Args too large for mailbox slot — fall through to pipe path */
    }

    /* ── Pipe fallback: for large payloads or when mailbox unavailable ─ */
    if (vm->cop_in_fd < 0) {
        /* Mailbox-only mode has no legacy pipes; fall back to in-process */
        return vm_ffi_call(module, import_idx, args, arg_count,
                           result, heap, error_msg, error_msg_size);
    }

    uint8_t payload[8192];
    uint32_t pos = 0;
    memcpy(payload + pos, &import_idx, 4); pos += 4;
    uint16_t argc = (uint16_t)arg_count;
    memcpy(payload + pos, &argc, 2); pos += 2;
    for (int i = 0; i < arg_count && i < 16; i++) {
        uint32_t n = cop_serialize_value(&args[i], payload + pos,
                                         sizeof(payload) - pos);
        if (n == 0) {
            snprintf(error_msg, error_msg_size, "COP: failed to serialize arg %d", i);
            return false;
        }
        pos += n;
    }

    if (!cop_send(vm->cop_in_fd, COP_MSG_FFI_REQ, payload, pos)) {
        vm_ffi_cop_stop(vm);
        snprintf(error_msg, error_msg_size,
                 "COP: pipe broken during FFI request");
        return false;
    }

    CopMsgHeader hdr;
    if (!cop_recv_header(vm->cop_out_fd, &hdr)) {
        vm_ffi_cop_stop(vm);
        snprintf(error_msg, error_msg_size,
                 "COP: pipe broken waiting for FFI response");
        return false;
    }

    if (hdr.msg_type == COP_MSG_FFI_RESULT) {
        if (hdr.payload_len > 0) {
            uint8_t *recv_buf = (hdr.payload_len <= sizeof(payload))
                                ? payload : malloc(hdr.payload_len);
            if (!recv_buf) {
                snprintf(error_msg, error_msg_size,
                         "COP: OOM for result (%u bytes)", hdr.payload_len);
                return false;
            }
            bool ok = cop_recv_payload(vm->cop_out_fd, recv_buf, hdr.payload_len);
            if (!ok) {
                if (recv_buf != payload) free(recv_buf);
                snprintf(error_msg, error_msg_size, "COP: failed to receive result");
                return false;
            }
            uint32_t consumed = cop_deserialize_value(recv_buf, hdr.payload_len,
                                                       result, heap);
            if (recv_buf != payload) free(recv_buf);
            if (consumed == 0) {
                snprintf(error_msg, error_msg_size, "COP: failed to deserialize result");
                return false;
            }
        } else {
            *result = val_void();
        }
        return true;
    }
    if (hdr.msg_type == COP_MSG_FFI_ERROR) {
        uint32_t elen = hdr.payload_len < (uint32_t)(error_msg_size - 1)
                        ? hdr.payload_len : (uint32_t)(error_msg_size - 1);
        if (elen > 0) { cop_recv_payload(vm->cop_out_fd, error_msg, elen); error_msg[elen] = '\0'; }
        return false;
    }
    snprintf(error_msg, error_msg_size, "COP: unexpected response 0x%02x", hdr.msg_type);
    return false;
}
