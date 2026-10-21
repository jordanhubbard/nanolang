#include "nanovm/heap.h"
#include <stdio.h>
#include <inttypes.h>
int g_argc=0;char **g_argv=NULL;
int main(void) {
    VmHeap heap;vm_heap_init(&heap);
    VmHashMap *map=vm_hashmap_new(&heap,TAG_STRING,TAG_INT);
    if(!map)return 1;
    uint32_t initial=map->bucket_count;
    for(unsigned i=0;i<64;i++) {
        char text[32];int length=snprintf(text,sizeof text,"key-%u",i);
        VmString *key=vm_string_new(&heap,text,(uint32_t)length);
        if(!key || !vm_hashmap_set(&heap,map,val_string(key),val_int(i)))return 2;
        vm_release(&heap,val_string(key));
    }
    uint32_t final=map->bucket_count;
    printf("buckets=%u->%u count=%u allocated=%zu freed-before=%zu objects=%zu\n",initial,final,map->count,heap.stats.allocated,heap.stats.freed,heap.stats.num_objects);
    vm_release(&heap,val_hashmap(map));vm_gc_collect_cycles(&heap);
    printf("allocated=%zu freed-after=%zu objects=%zu expected-growth=%zu\n",heap.stats.allocated,heap.stats.freed,heap.stats.num_objects,(size_t)(final-initial)*sizeof(VmHMEntry));
    int result=heap.stats.num_objects?3:0;vm_heap_destroy(&heap);return result;
}
