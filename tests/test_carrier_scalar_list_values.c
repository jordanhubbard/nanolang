/* I retain explicit scalar headers without interpreting their cells as roots. */
#include "eval/carrier_values.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"I fail %s at line %d.\n",#c,__LINE__);exit(1); } } while (0)
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
static void owned(void) {
    NlCarrierValues *heap=nl_carrier_values_new();CHECK(heap);
    Value record=create_struct("ScalarBits",NULL,NULL,0);
    CHECK(nl_carrier_values_adopt(heap,nl_carrier_value(record)));
    List_int *list=list_int_new();list_int_push(list,17);list_int_push(list,-3);
    list_int_push(list,(intptr_t)record.as.struct_val);
    NlCarrierRef scalar=nl_carrier_legacy_scalar_list(list),logical=scalar;
    logical.record_list=NL_CARRIER_RECORD_LIST;
    CHECK(nl_carrier_values_adopt(heap,scalar));
    CHECK(nl_carrier_values_collect(heap,&logical,1));
    CHECK(nl_carrier_values_stats(heap).live==1 && nl_carrier_values_stats(heap).destroyed==1);
    CHECK(list_int_get(list,0)==17 && list_int_get(list,1)==-3);
    NlCarrierRef wrong=nl_carrier_legacy_record_list(list);
    CHECK(!nl_carrier_values_collect(heap,&wrong,1));
    CHECK(nl_carrier_values_stats(heap).live==1);
    CHECK(nl_carrier_values_consume_legacy_list(heap,list));
    CHECK(!nl_carrier_values_consume_legacy_list(heap,list));
    CHECK(nl_carrier_values_collect(heap,&logical,1));
    CHECK(nl_carrier_values_consumed_count(heap)==1);
    CHECK(nl_carrier_values_collect(heap,NULL,0));
    CHECK(nl_carrier_values_consumed_count(heap)==0);
    list=list_int_new();list_int_push(list,-3);scalar=nl_carrier_legacy_scalar_list(list);
    CHECK(nl_carrier_values_adopt(heap,scalar));
    CHECK(nl_carrier_values_transfer(heap,&scalar,1));
    CHECK(nl_carrier_values_stats(heap).transferred==1);
    CHECK(nl_carrier_values_free(heap));CHECK(list_int_get(list,0)==-3);list_int_free(list);
}
static void borrowed(void) {
    NlCarrierValues *heap=nl_carrier_values_new();CHECK(heap);
    List_int *list=list_int_new();list_int_push(list,17);list_int_push(list,-3);
    NlCarrierRef roots[]={nl_carrier_legacy_scalar_list(list),nl_carrier_legacy_record_list(list)};
    CHECK(nl_carrier_values_begin_list_borrow(heap,roots[0]));
    CHECK(nl_carrier_values_collect(heap,roots,1));
    CHECK(!nl_carrier_values_collect(heap,roots,2));
    NlCarrierRef reverse[]={roots[1],roots[0]};
    CHECK(!nl_carrier_values_collect(heap,reverse,2));
    CHECK(nl_carrier_values_stats(heap).registered==0);
    CHECK(nl_carrier_values_collect(heap,roots,1));
    roots[1]=roots[0];roots[1].record_list=255;
    CHECK(!nl_carrier_values_collect(heap,roots+1,1));
    int capacity=list->capacity;list->capacity=0;
    CHECK(!nl_carrier_values_collect(heap,roots,1));list->capacity=capacity;
    CHECK(nl_carrier_values_collect(heap,roots,1));
    DynArray *managed=dyn_array_new(ELEM_INT);CHECK(managed);
    NlCarrierRef bad=nl_carrier_legacy_scalar_list((List_int *)managed);
    CHECK(!nl_carrier_values_adopt(heap,bad));
    CHECK(!nl_carrier_values_begin_list_borrow(heap,bad));gc_release(managed);
    CHECK(nl_carrier_values_free(heap));CHECK(list_int_get(list,0)==17);list_int_free(list);
}
int main(void) {
    owned();borrowed();CHECK(gc_get_stats().num_objects==0);
    puts("I preserve scalar-bit nonownership, logical roots, immediate free, transfer, caller storage and conflicting-provenance refusals.");
    return 0;
}
