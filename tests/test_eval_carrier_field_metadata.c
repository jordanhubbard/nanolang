/* I borrow field declarations for a walk without caching payloads or owners. */
#include "eval/carrier_values.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"I fail %s at line %d.\n",#c,__LINE__);exit(1); } } while (0)
int g_argc;
char **g_argv;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }
typedef struct {
    NlCarrierValues *heap;
    char *names[2];Type types[2];size_t resolved,old_calls;
} Context;
static bool holder(Value value) {
    return (value.type==VAL_STRUCT && !strcmp(value.as.struct_val->struct_name,"FieldHolder")) ||
        (value.type==VAL_UNION && !strcmp(value.as.union_val->union_name,"FieldUnion"));
}
static bool old_slot(Value owner,int index,void *opaque) {
    Context *ctx=opaque;ctx->old_calls++;return holder(owner) && index==1;
}
static NlCarrierRecordListFields fields(Value owner,void *opaque) {
    Context *ctx=opaque;ctx->resolved++;
    CHECK(!nl_carrier_values_clear(ctx->heap));CHECK(!nl_carrier_values_free(ctx->heap));
    nl_carrier_values_set_slots(ctx->heap,old_slot,ctx);
    return holder(owner) ? (NlCarrierRecordListFields){ctx->names,ctx->types,2} : (NlCarrierRecordListFields){0};
}
static Value child(NlCarrierValues *heap) {
    Value text=create_string("kept");char *name="text";
    Value result=create_struct("FieldItem",&name,&text,1);gc_release(text.as.string_val);
    CHECK(nl_carrier_values_adopt(heap,nl_carrier_value(result)));return result;
}
static Value list(NlCarrierValues *heap,Value item,bool owned) {
    if(owned) {
        List_int *header=list_int_new();list_int_push(header,(intptr_t)item.as.struct_val);
        CHECK(nl_carrier_values_adopt(heap,nl_carrier_legacy_record_list(header)));
        return create_int((intptr_t)header);
    }
    DynArray *header=dyn_array_new(ELEM_INT);CHECK(header);
    dyn_array_push_int(header,(intptr_t)item.as.struct_val);return create_int((intptr_t)header);
}
static void release_borrowed(Value value,bool owned) { if(!owned)gc_release((void *)(intptr_t)value.as.int_val); }
static void check(bool owned,bool union_owner) {
    NlCarrierValues *heap=nl_carrier_values_new();CHECK(heap);
    Context ctx={.heap=heap,.names={"items","bits"},.types={TYPE_LIST_GENERIC,TYPE_INT}};
    nl_carrier_values_set_fields(heap,fields,&ctx);
    Value item=child(heap),encoded=list(heap,item,owned),inputs[]={encoded,encoded};
    char *names[]={"bits","items"};
    Value parent=union_owner ? create_union("FieldUnion",0,"Some",names,inputs,2) : create_struct("FieldHolder",names,inputs,2);
    CHECK(nl_carrier_values_adopt(heap,nl_carrier_value(parent)));
    Value *values=union_owner ? parent.as.union_val->field_values : parent.as.struct_val->field_values;
    NlCarrierRef root=nl_carrier_value(parent);size_t expected=owned ? 3u : 2u;
    CHECK(nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==expected);
    CHECK(!strcmp(item.as.struct_val->field_values[0].as.string_val,"kept"));CHECK(ctx.old_calls==0);
    values[1]=create_bool(true);CHECK(!nl_carrier_values_collect(heap,&root,1));
    CHECK(nl_carrier_values_stats(heap).live==expected);values[1]=encoded;
    CHECK(nl_carrier_values_collect(heap,&root,1));
    ctx.types[0]=TYPE_INT;
    CHECK(nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==1);
    release_borrowed(encoded,owned);values[1]=create_int(0);
    item=child(heap);encoded=list(heap,item,owned);values[1]=encoded;ctx.types[0]=TYPE_LIST_GENERIC;
    CHECK(nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==expected);
    size_t resolved=ctx.resolved;ctx.types[0]=TYPE_INT;
    nl_carrier_values_set_slots(heap,old_slot,&ctx);
    CHECK(nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==expected);
    CHECK(ctx.resolved==resolved && ctx.old_calls>0);
    nl_carrier_values_set_fields(heap,fields,&ctx);
    CHECK(nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==1);
    release_borrowed(encoded,owned);values[1]=create_int(0);
    ctx.names[0]="bits";ctx.names[1]="items";ctx.types[0]=TYPE_INT;ctx.types[1]=TYPE_LIST_GENERIC;
    item=child(heap);encoded=list(heap,item,owned);values[1]=encoded;
    CHECK(nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==expected);
    char ***owner_names=union_owner ? &parent.as.union_val->field_names : &parent.as.struct_val->field_names;
    char **saved=*owner_names;*owner_names=NULL;
    CHECK(!nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==expected);
    *owner_names=saved;CHECK(nl_carrier_values_collect(heap,&root,1));
    ctx.names[0]="items";ctx.names[1]="items";
    CHECK(nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==1);
    release_borrowed(encoded,owned);values[1]=create_int(0);
    ctx.names[0]="items";ctx.names[1]="bits";ctx.types[0]=TYPE_LIST_GENERIC;ctx.types[1]=TYPE_INT;
    item=child(heap);encoded=list(heap,item,owned);values[1]=encoded;
    CHECK(nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==expected);
    CHECK(nl_carrier_values_free(heap));
    if(!owned)CHECK(((DynArray *)(intptr_t)encoded.as.int_val)->length==1);
    release_borrowed(encoded,owned);
    CHECK(gc_get_stats().num_objects==0 && gc_get_stats().current_usage==0);
}
static NlCarrierRecordListFields large_fields(Value owner,void *context) {
    (void)owner;return *(NlCarrierRecordListFields *)context;
}
static void large_schema(int count) {
    char storage[129][32];char *names[129];Type types[129];
    for(int i=0;i<count;i++) {
        snprintf(storage[i],sizeof storage[i],"unused_%d",i);names[i]=storage[i];types[i]=TYPE_INT;
    }
    /* I require a real FNV low-byte collision with items (bucket143). */
    names[0]="collision_351";
    names[count-1]="items";types[count-1]=TYPE_LIST_GENERIC;
    NlCarrierRecordListFields schema={names,types,count};
    NlCarrierValues *heap=nl_carrier_values_new();CHECK(heap);
    nl_carrier_values_set_fields(heap,large_fields,&schema);
    Value item=child(heap),encoded=list(heap,item,true);char *name="items";
    Value parent=create_struct("LargeFields",&name,&encoded,1);
    CHECK(nl_carrier_values_adopt(heap,nl_carrier_value(parent)));
    NlCarrierRef root=nl_carrier_value(parent);
    CHECK(nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==3);
    CHECK(!strcmp(item.as.struct_val->field_values[0].as.string_val,"kept"));
    /* I read changed first-name metadata on both sides of the stack boundary. */
    names[0]="items";types[0]=TYPE_INT;
    CHECK(nl_carrier_values_collect(heap,&root,1));CHECK(nl_carrier_values_stats(heap).live==1);
    parent.as.struct_val->field_values[0]=create_int(0);
    CHECK(nl_carrier_values_free(heap));CHECK(gc_get_stats().num_objects==0);
}
int main(void) {
    for(int count=127;count<=129;count++)large_schema(count);
    check(false,false);check(false,true);check(true,false);check(true,true);
    puts("I preserve changed/reordered/first-match field metadata, legacy and borrowed canonical ownership, union/record fields, old slot callbacks, reentry and failed-trace recovery.");
    return 0;
}
