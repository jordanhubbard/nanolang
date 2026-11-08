#define _POSIX_C_SOURCE 200809L
#include "nanolang.h"
#include "eval/eval_io.h"
#include "runtime/dyn_array.h"
#include "runtime/gc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern DynArray *vm_bytes_from_string(const char *);
static DynArray* nl_bytes_from_string(const char* s) {
    DynArray* out = dyn_array_new(ELEM_U8);
    if (!out) return NULL;
    if (!s) return out;
    size_t len = strnlen(s, 64*1024*1024);
    for (size_t i = 0; i < len; i++) {
        dyn_array_push_u8(out, (uint8_t)(unsigned char)s[i]);
    }
    return out;
}


int main(void) {
 const size_t cap = (size_t)64 * 1024 * 1024;
 const size_t sizes[] = {cap-1, cap, cap+17};
 char *text = malloc(cap+18); assert(text);
 gc_init();
 for (size_t index=0;index<3;++index) {
  size_t length=sizes[index]; memset(text,'A',length);text[length-1]='Z';text[length]=0;
  assert(strlen(text)==length);
  DynArray *generated=nl_bytes_from_string(text);assert(generated);
  size_t encoded=length>cap?cap:length;
  assert(dyn_array_get_elem_type(generated)==ELEM_U8);
  assert((size_t)dyn_array_length(generated)==encoded);
  assert(dyn_array_get_u8(generated,(int64_t)encoded-1)==(length>cap?'A':'Z'));
  printf("input=%zu emitted_c_helper=%lld emitted_last=%u ",length,(long long)dyn_array_length(generated),dyn_array_get_u8(generated,(int64_t)encoded-1));
  gc_release(generated);
  Value argument={0};argument.type=VAL_STRING;argument.as.string_val=text;
  Value interpreted=builtin_bytes_from_string(&argument);assert(interpreted.type==VAL_DYN_ARRAY);
  assert(dyn_array_get_elem_type(interpreted.as.dyn_array_val)==ELEM_U8);
  assert((size_t)dyn_array_length(interpreted.as.dyn_array_val)==length);
  assert(dyn_array_get_u8(interpreted.as.dyn_array_val,(int64_t)length-1)=='Z');
  printf("interpreter_builtin=%lld ",(long long)dyn_array_length(interpreted.as.dyn_array_val));
  gc_release(interpreted.as.dyn_array_val);
  DynArray *legacy_vm=vm_bytes_from_string(text);assert(legacy_vm);
  assert(dyn_array_get_elem_type(legacy_vm)==ELEM_INT);
  assert((size_t)dyn_array_length(legacy_vm)==length);
  assert(dyn_array_get_int(legacy_vm,(int64_t)length-1)=='Z');
  printf("legacy_vm_c_builtin=%lld\n",(long long)dyn_array_length(legacy_vm));
  gc_release(legacy_vm);
 }
 free(text);gc_shutdown();
 assert(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
 return 0;
}
