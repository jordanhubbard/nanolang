/* My O0 fixture aggregates may lower to memory helpers. I keep them import-free. */
#ifndef MIXED_COUNTED_WASM_MEMORY_H
#define MIXED_COUNTED_WASM_MEMORY_H
#ifdef __wasm32__
#include <stddef.h>
#include <stdint.h>
void *memcpy(void *destination,const void *source,size_t count){
    volatile unsigned char *d=destination;const volatile unsigned char *s=source;
    for(size_t i=0;i<count;i++)d[i]=s[i];
    return destination;
}
void *memset(void *destination,int value,size_t count){
    volatile unsigned char *d=destination;
    for(size_t i=0;i<count;i++)d[i]=(unsigned char)value;
    return destination;
}
void *memmove(void *destination,const void *source,size_t count){
    volatile unsigned char *d=destination;const volatile unsigned char *s=source;
    if((uintptr_t)d<(uintptr_t)s){for(size_t i=0;i<count;i++)d[i]=s[i];}
    else {while(count){count--;d[count]=s[count];}}
    return destination;
}
#endif
#endif
