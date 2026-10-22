/* This harness calls only copied observation APIs for its semantic oracle. */
#define MAP_RUNTIME_MAIN retained_map_runtime_controls
#include "test_record_map_runtime.c"
static VmRmObservationWork observation_work(void) {
    VmRmObservationWork w={.limit=VM_RM_OBSERVATION_WORK_MAX};return w;
}

typedef struct {VmRecordMapPrivate *p;VmRecordMapObservation out;VmRmObservationWork work;VmRecordMapObservationStatus status;} ObservationThread;
static void *observation_other_thread(void *arg) {
    ObservationThread *t=arg;t->status=vm_record_map_private_observe(t->p,VM_RM_RESULT,0,NULL,0,&t->work,&t->out);return NULL;
}
static void observation_refusals(VmRecordMapPrivate *p) {
    VmRecordMapObservation sentinel,out;memset(&sentinel,0xa5,sizeof sentinel);VmRmObservationWork w;
    for(unsigned invalid=0;invalid<5;invalid++) {
        if(invalid==0)p->busy=true;
        if(invalid==1)p->vm.heap.gc_running=true;
        if(invalid==2)p->vm.activation_floor=1;
        if(invalid==3)p->vm.references.active=true;
        if(invalid==4)p->vm.reference_generation=1;
        out=sentinel;w=observation_work();CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&w,&out)==VM_RM_OBS_INVALID);
        CHECK(!memcmp(&out,&sentinel,sizeof out)&&w.used==16);
        p->busy=false;p->vm.heap.gc_running=false;p->vm.activation_floor=0;p->vm.references.active=false;p->vm.reference_generation=0;
    }
    ObservationThread t={.p=p,.out=sentinel,.work=observation_work()};pthread_t thread;
    CHECK(!pthread_create(&thread,NULL,observation_other_thread,&t));CHECK(!pthread_join(thread,NULL));
    CHECK(t.status==VM_RM_OBS_INVALID&&!memcmp(&t.out,&sentinel,sizeof sentinel)&&t.work.used==16);
    out=sentinel;w=observation_work();w.limit=32;CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&w,&out)==VM_RM_OBS_LIMIT&&!memcmp(&out,&sentinel,sizeof out));
    w=observation_work();w.limit=33;CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&w,&out)==VM_RM_OBS_OK&&w.used==33);
    out=sentinel;w=observation_work();CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,(uint32_t *)&out,1,&w,&out)==VM_RM_OBS_INVALID&&w.used==0&&!memcmp(&out,&sentinel,sizeof out));
    uint32_t field=0;w=observation_work();CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&field,1,&out,1,&w,&out)==VM_RM_OBS_INVALID&&w.used==0&&!memcmp(&out,&sentinel,sizeof out));
    w=observation_work();CHECK(vm_record_map_private_string(p,VM_RM_GLOBAL,2,NULL,0,0,2,&w,(void *)(UINTPTR_MAX-1))==VM_RM_OBS_INVALID&&w.used==0);
    w=observation_work();CHECK(vm_record_map_private_string(p,VM_RM_GLOBAL,2,NULL,0,4,0,&w,NULL)==VM_RM_OBS_OK&&w.used==34);
    VmRecordMapIdentityComparison same,prior;memset(&prior,0xa5,sizeof prior);w=observation_work();
    CHECK(vm_record_map_private_same_identity(p,VM_RM_RESULT,0,&field,1,VM_RM_GLOBAL,0,&field,1,&w,&same)==VM_RM_OBS_OK);uint64_t exact=w.used;
    same=prior;w=observation_work();w.limit=exact-1;
    CHECK(vm_record_map_private_same_identity(p,VM_RM_RESULT,0,&field,1,VM_RM_GLOBAL,0,&field,1,&w,&same)==VM_RM_OBS_LIMIT&&!memcmp(&same,&prior,sizeof same));
    w=observation_work();w.limit=exact;CHECK(vm_record_map_private_same_identity(p,VM_RM_RESULT,0,&field,1,VM_RM_GLOBAL,0,&field,1,&w,&same)==VM_RM_OBS_OK&&w.used==exact);
    printf("I measured exact observe33/string36/empty-string34/identity%llu logical work.\n",(unsigned long long)exact);
}
static void observation_topology(VmRecordMapPrivate *p) {
    VmHashMap *m=p->result.as.sval->fields[0].as.hashmap;uint32_t field=0,occupied=UINT32_MAX,empty=UINT32_MAX;
    for(uint32_t i=0;i<m->bucket_count;i++){if(m->entries[i].state)occupied=i;else empty=i;}
    CHECK(occupied!=UINT32_MAX&&empty!=UINT32_MAX);VmHMEntry saved=m->entries[occupied];uint32_t count=m->count,hash=saved.key.as.string->hash;
    VmRecordMapObservation out,sentinel;memset(&sentinel,0xa5,sizeof sentinel);
    for(unsigned bad=0;bad<4;bad++) {
        if(bad==0)m->count++;
        if(bad==1)saved.key.as.string->hash^=1;
        if(bad==2){m->entries[empty]=saved;memset(&m->entries[occupied],0,sizeof saved);}
        if(bad==3){m->entries[empty]=saved;m->count++;vm_retain(&p->vm.heap,saved.key);}
        out=sentinel;VmRmObservationWork w=observation_work();CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&field,1,"leaf",4,&w,&out)==VM_RM_OBS_INVALID&&!memcmp(&out,&sentinel,sizeof out));
        m->count=count;saved.key.as.string->hash=hash;m->entries[occupied]=saved;memset(&m->entries[empty],0,sizeof saved);
        if(bad==3)vm_release(&p->vm.heap,saved.key);
        w=observation_work();CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&field,1,"leaf",4,&w,&out)==VM_RM_OBS_OK&&out.scalar_bits==37);
    }
    /* Genuine counted keys and a long collision chain, using the checked
     * adapter to establish valid topology; no guest admission is inferred. */
    unsigned found=0;char last[32]={0};
    for(unsigned candidate=0;found<64;candidate++) {
        char text[32];int n=snprintf(text,sizeof text,"collision%u",candidate);CHECK(n>0&&(size_t)n<sizeof text);
        VmString *key=vm_string_new(&p->vm.heap,text,(uint32_t)n);CHECK(key);
        if(key->hash%128==0){CHECK(vm_rm_heap_set(&p->vm.heap,m,val_string(key),val_int(found))==VM_RM_HEAP_OK);found++;memcpy(last,text,(size_t)n+1);}
        vm_release(&p->vm.heap,val_string(key));
    }
    const char nul_keys[2][3]={{'a',0,'x'},{'a',0,'y'}};
    for(unsigned i=0;i<2;i++){VmString *key=vm_string_new(&p->vm.heap,nul_keys[i],3);CHECK(key);CHECK(vm_rm_heap_set(&p->vm.heap,m,val_string(key),val_int(80+i))==VM_RM_HEAP_OK);vm_release(&p->vm.heap,val_string(key));}
    size_t calls=ra_calls,live=ra_live,bytes=ra_bytes;VmRmObservationWork w=observation_work();
    CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&field,1,last,(uint32_t)strlen(last),&w,&out)==VM_RM_OBS_OK&&out.scalar_bits==63);
    uint64_t exact=w.used;out=sentinel;w=observation_work();w.limit=exact-1;
    CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&field,1,last,(uint32_t)strlen(last),&w,&out)==VM_RM_OBS_LIMIT&&!memcmp(&out,&sentinel,sizeof out));
    for(unsigned i=0;i<2;i++){w=observation_work();CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&field,1,nul_keys[i],3,&w,&out)==VM_RM_OBS_OK&&out.scalar_bits==80+i);}
    CHECK(ra_calls==calls&&ra_live==live&&ra_bytes==bytes);printf("I bounded64 colliding keys at%llu lookup work and retained exact counted-NUL keys.\n",(unsigned long long)exact);
}
static void observation_basics(void) {
    Input c;init(&c,TAG_INT,false);c.fn[0].result_tag=TAG_STRUCT;
    map_new(&c,TAG_STRING,TAG_INT);scalar(&c,TAG_STRING);runtime_integer(&c,37);op(&c,OP_HM_SET);
    record(&c,0);op(&c,OP_DUP);arg32(&c,OP_STORE_GLOBAL,0);arg16(&c,OP_STORE_LOCAL,0);
    map_new(&c,TAG_STRING,TAG_INT);record(&c,0);arg32(&c,OP_STORE_GLOBAL,1);
    scalar(&c,TAG_STRING);arg32(&c,OP_STORE_GLOBAL,2);
    arg16(&c,OP_LOAD_LOCAL,0);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;map_authority(&c);
    VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
    size_t calls=ra_calls,live=ra_live,bytes=ra_bytes;uint32_t path=0;
    VmRecordMapObservation result;memset(&result,0xa5,sizeof result);VmRmObservationWork w=observation_work();
    CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&w,&result)==VM_RM_OBS_OK);
    CHECK(result.tag==TAG_STRUCT&&result.layout==0&&result.length==1&&result.epoch==1&&w.used==33);
    w=observation_work();CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,&path,1,&w,&result)==VM_RM_OBS_OK);
    CHECK(result.tag==TAG_HASHMAP&&result.length==1&&result.key_tag==TAG_STRING&&result.value_tag==TAG_INT);
    w=observation_work();CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&path,1,"leaf",4,&w,&result)==VM_RM_OBS_OK);
    CHECK(result.tag==TAG_INT&&result.scalar_bits==37);uint64_t lookup_work=w.used;
    VmRecordMapObservation sentinel;memset(&sentinel,0xa5,sizeof sentinel);result=sentinel;w=observation_work();w.limit=lookup_work-1;
    CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&path,1,"leaf",4,&w,&result)==VM_RM_OBS_LIMIT);
    CHECK(w.limited&&!memcmp(&result,&sentinel,sizeof result));
    w=observation_work();w.limit=lookup_work;CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&path,1,"leaf",4,&w,&result)==VM_RM_OBS_OK&&w.used==lookup_work);
    w=observation_work();CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&path,1,"missing",7,&w,&result)==VM_RM_OBS_OK&&result.tag==TAG_VOID&&result.scalar_bits==0);
    VmRecordMapIdentityComparison same;w=observation_work();
    CHECK(vm_record_map_private_same_identity(p,VM_RM_RESULT,0,&path,1,VM_RM_GLOBAL,0,&path,1,&w,&same)==VM_RM_OBS_OK&&same.same&&same.epoch==1);
    w=observation_work();CHECK(vm_record_map_private_same_identity(p,VM_RM_RESULT,0,&path,1,VM_RM_GLOBAL,1,&path,1,&w,&same)==VM_RM_OBS_OK&&!same.same);
    char output[4]={'!','!','!','!'};w=observation_work();
    CHECK(vm_record_map_private_string(p,VM_RM_GLOBAL,2,NULL,0,1,2,&w,output)==VM_RM_OBS_OK);
    CHECK(!memcmp(output,"ea!!",4)&&w.used==36);memset(output,'!',sizeof output);w=observation_work();w.limit=35;
    CHECK(vm_record_map_private_string(p,VM_RM_GLOBAL,2,NULL,0,1,2,&w,output)==VM_RM_OBS_LIMIT&&!memcmp(output,"!!!!",4));
    result=sentinel;w=observation_work();w.limited=true;
    CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&w,&result)==VM_RM_OBS_LIMIT&&w.used==0&&!memcmp(&result,&sentinel,sizeof result));
    w.used=w.limit+1;CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&w,&result)==VM_RM_OBS_INVALID&&!memcmp(&result,&sentinel,sizeof result));
    union {VmRmObservationWork work;VmRecordMapObservation result;} overlap;memset(&overlap,0,sizeof overlap);overlap.work.limit=1000;
    unsigned char saved[sizeof overlap];memcpy(saved,&overlap,sizeof overlap);
    CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&overlap.work,&overlap.result)==VM_RM_OBS_INVALID&&!memcmp(saved,&overlap,sizeof overlap));
    w=observation_work();CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,&path,259,&w,&result)==VM_RM_OBS_INVALID&&w.used==0);
    CHECK(ra_calls==calls&&ra_live==live&&ra_bytes==bytes);
    printf("I copied observations%zu/identity%zu with owner%zu unchanged; lookup work%llu.\n",sizeof result,sizeof same,sizeof *p,(unsigned long long)lookup_work);
    observation_refusals(p);observation_topology(p);
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}

static void observation_tree(Input *c,unsigned records,unsigned depth,uint8_t leaf) {
    init(c,TAG_INT,false);c->m.struct_count=records;c->fn[0].result_tag=TAG_STRUCT;
    for(unsigned i=1;i<depth;i++)array(c,TAG_ARRAY);
    array(c,leaf);scalar(c,leaf);op(c,OP_ARR_PUSH);
    for(unsigned i=1;i<depth;i++)op(c,OP_ARR_PUSH);
    map_new(c,TAG_STRING,TAG_INT);scalar(c,TAG_STRING);runtime_integer(c,91);op(c,OP_HM_SET);
    op(c,OP_STRUCT_LITERAL);b32(c->code,&c->n,0);b16(c->code,&c->n,2);
    for(unsigned i=1;i<records;i++)record(c,i);
    op(c,OP_RET);c->fn[0].code_length=(uint32_t)c->n;
    b32(c->layouts,&c->l,records);
    for(unsigned i=0;i<records;i++) {
        b8(c->layouts,&c->l,NVM_V2_LAYOUT_STRUCT);b8(c->layouts,&c->l,0);b16(c->layouts,&c->l,i?1:2);b32(c->layouts,&c->l,NO);
        for(unsigned field=0;field<(i?1u:2u);field++) {
            b8(c->layouts,&c->l,i?TAG_STRUCT:field?TAG_HASHMAP:TAG_ARRAY);b8(c->layouts,&c->l,0);b16(c->layouts,&c->l,0);
            b32(c->layouts,&c->l,i?i-1:NO);b32(c->layouts,&c->l,NO);
        }
    }
    b32(c->ownership,&c->o,3);b32(c->ownership,&c->o,records);
    for(unsigned i=0;i<records;i++)b8(c->ownership,&c->o,NVM_LAYOUT_COMPLETE);
    while(c->o%4)b8(c->ownership,&c->o,0);
    b32(c->ownership,&c->o,1);b16(c->ownership,&c->o,c->fn[0].local_count);b16(c->ownership,&c->o,0);desc(c,TAG_STRUCT);
    for(unsigned i=0;i<c->fn[0].local_count;i++)desc(c,TAG_VOID);
    b32(c->ownership,&c->o,4);b32(c->ownership,&c->o,0);b32(c->ownership,&c->o,2);
    b16(c->ownership,&c->o,2);b16(c->ownership,&c->o,1);b32(c->ownership,&c->o,20+depth*8);b32(c->ownership,&c->o,depth);
    for(unsigned i=0;i<depth;i++){b8(c->ownership,&c->o,i+1==depth?leaf:TAG_ARRAY);b8(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,i+1==depth?NO:i+1);}
    b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,0);
    b16(c->ownership,&c->o,5);b16(c->ownership,&c->o,1);b32(c->ownership,&c->o,48);b32(c->ownership,&c->o,2);desc(c,TAG_INT);desc(c,TAG_STRING);
    b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,1);b32(c->ownership,&c->o,0);b32(c->ownership,&c->o,1);
    b32(c->ownership,&c->o,0);b16(c->ownership,&c->o,1);b16(c->ownership,&c->o,0);b32(c->ownership,&c->o,0);
    c->m.layout_size=(uint32_t)c->l;c->m.ownership_size=(uint32_t)c->o;c->m.code_size=(uint32_t)c->n;
}
static void observation_selected_tree(void) {
    const uint8_t leaves[]={TAG_INT,TAG_U8,TAG_FLOAT,TAG_BOOL,TAG_STRING};
    for(unsigned leaf=0;leaf<5;leaf++)for(unsigned depth=1;depth<=3;depth++) {
        Input c;observation_tree(&c,2,depth,leaves[leaf]);VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
        uint32_t path[5]={0};VmRecordMapObservation out,sentinel;memset(&sentinel,0xa5,sizeof sentinel);VmRmObservationWork w=observation_work();
        CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,path,(uint16_t)(depth+2),&w,&out)==VM_RM_OBS_OK&&out.tag==leaves[leaf]);
        /* Only the selected path is independently checked, not every sibling. */
        VmStruct *bottom=p->result.as.sval->fields[0].as.sval;NanoValue saved=bottom->fields[1];bottom->fields[1]=val_int(9);
        w=observation_work();CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,path,(uint16_t)(depth+2),&w,&out)==VM_RM_OBS_OK);
        path[1]=1;out=sentinel;w=observation_work();CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,path,2,&w,&out)==VM_RM_OBS_INVALID&&!memcmp(&out,&sentinel,sizeof out));
        bottom->fields[1]=saved;path[1]=0;
        NanoValue saved_array=bottom->fields[0];VmArray *wrong=vm_array_new(&p->vm.heap,depth==1?TAG_ARRAY:leaves[leaf],0);CHECK(wrong);
        bottom->fields[0]=val_array(wrong);out=sentinel;w=observation_work();
        CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,path,2,&w,&out)==VM_RM_OBS_INVALID&&!memcmp(&out,&sentinel,sizeof out));
        bottom->fields[0]=saved_array;vm_release(&p->vm.heap,val_array(wrong));
        vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
    }
    Input c;observation_tree(&c,60,3,TAG_BOOL);VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
    uint32_t path[259]={0};VmRecordMapObservation out;VmRmObservationWork w=observation_work();
    CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,path,63,&w,&out)==VM_RM_OBS_OK&&out.tag==TAG_BOOL&&out.scalar_bits==1);
    w=observation_work();CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,path,259,&w,&out)==VM_RM_OBS_INVALID&&w.used==0);
    printf("I observed actual64-origin tree scalar path63; path259 refused without traversal.\n");
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static void observation_failed_epoch(void) {
    Input c;init(&c,TAG_INT,false);c.fn[0].result_tag=TAG_STRUCT;
    arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_TYPE_CHECK);b8(c.code,&c.n,TAG_VOID);
    size_t initialized=c.n;arg32(&c,OP_JMP_FALSE,0);
    map_new(&c,TAG_STRING,TAG_INT);scalar(&c,TAG_STRING);runtime_integer(&c,0);op(&c,OP_HM_SET);
    record(&c,0);arg32(&c,OP_STORE_GLOBAL,0);runtime_patch(&c,initialized);
    arg32(&c,OP_LOAD_GLOBAL,0);arg16(&c,OP_STRUCT_GET,0);arg16(&c,OP_STORE_LOCAL,0);
    arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,TAG_STRING);op(&c,OP_HM_GET);op(&c,OP_CAST_INT);
    runtime_integer(&c,1);op(&c,OP_I64_ADD);arg16(&c,OP_STORE_LOCAL,1);
    arg16(&c,OP_LOAD_LOCAL,0);scalar(&c,TAG_STRING);arg16(&c,OP_LOAD_LOCAL,1);op(&c,OP_HM_SET);op(&c,OP_POP);
    arg16(&c,OP_LOAD_LOCAL,1);runtime_integer(&c,2);op(&c,OP_I64_LT_S);op(&c,OP_ASSERT);
    arg32(&c,OP_LOAD_GLOBAL,0);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;map_authority(&c);

    VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
    VmRecordMapObservation first,second;VmRmObservationWork w=observation_work();
    CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&w,&first)==VM_RM_OBS_OK&&first.epoch==1&&first.tag==TAG_STRUCT);
    CHECK(vm_record_map_private_run(p)==VM_ERR_ASSERT_FAILED);
    w=observation_work();CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&w,&second)==VM_RM_OBS_OK&&second.epoch==2&&second.layout==first.layout);
    uint32_t field=0;VmRecordMapIdentityComparison same;
    CHECK(vm_record_map_private_same_identity(p,VM_RM_RESULT,0,&field,1,VM_RM_GLOBAL,0,&field,1,&w,&same)==VM_RM_OBS_OK&&same.same&&same.epoch==2);
    CHECK(vm_record_map_private_lookup(p,VM_RM_RESULT,0,&field,1,"leaf",4,&w,&second)==VM_RM_OBS_OK&&second.tag==TAG_INT&&second.scalar_bits==2);
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
static void observation_empty_root(void) {
    Input c;observation_tree(&c,2,3,TAG_BOOL);CHECK(c.code[c.n-1]==OP_RET);c.n--;op(&c,OP_POP);
    const unsigned char payload[]={0,'"','\\',0xc3,0xa9,'\n',0};c.strings[1]=(char *)payload;c.lengths[1]=6;
    scalar(&c,TAG_STRING);arg32(&c,OP_STORE_GLOBAL,1);
    op(&c,OP_PUSH_F64);for(unsigned i=0;i<8;i++)b8(c.code,&c.n,i==7?0x80:0);arg32(&c,OP_STORE_GLOBAL,0);
    array(&c,TAG_ARRAY);op(&c,OP_RET);c.fn[0].code_length=(uint32_t)c.n;c.m.code_size=(uint32_t)c.n;
    CHECK(c.ownership[20]==TAG_STRUCT);c.ownership[20]=TAG_ARRAY;c.fn[0].result_tag=TAG_ARRAY;
    VmRecordMapPrivate *p=runtime_instance(&c);CHECK(vm_record_map_private_run(p)==VM_OK);
    VmRecordMapObservation out;VmRmObservationWork w=observation_work();
    CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&w,&out)==VM_RM_OBS_OK&&out.tag==TAG_ARRAY&&out.element_tag==TAG_ARRAY&&out.length==0);
    w=observation_work();CHECK(vm_record_map_private_observe(p,VM_RM_GLOBAL,0,NULL,0,&w,&out)==VM_RM_OBS_OK&&out.tag==TAG_FLOAT&&out.scalar_bits==UINT64_C(0x8000000000000000));
    unsigned char bytes[6]={0};w=observation_work();
    CHECK(vm_record_map_private_string(p,VM_RM_GLOBAL,1,NULL,0,0,6,&w,bytes)==VM_RM_OBS_OK&&!memcmp(bytes,payload,6)&&w.used==40);
    memset(bytes,0xa5,sizeof bytes);w=observation_work();w.limit=39;
    CHECK(vm_record_map_private_string(p,VM_RM_GLOBAL,1,NULL,0,0,6,&w,bytes)==VM_RM_OBS_LIMIT);
    for(unsigned i=0;i<sizeof bytes;i++)CHECK(bytes[i]==0xa5);
    vm_record_map_private_destroy(p);CHECK(!ra_live&&!ra_bytes);
}
int main(int argc,char **argv) {
    observation_basics();observation_selected_tree();observation_failed_epoch();observation_empty_root();
    for(int i=1;i<argc;i++)map_original_module(argv[i]);
    for(unsigned i=0;i<original_count;i++){VmRecordMapPrivate *p=originals[i];CHECK(vm_record_map_private_run(p)==VM_OK);VmRmObservationWork w=observation_work();VmRecordMapObservation out;CHECK(vm_record_map_private_observe(p,VM_RM_RESULT,0,NULL,0,&w,&out)==VM_RM_OBS_OK&&out.tag==TAG_INT&&out.scalar_bits==0);vm_record_map_private_destroy(p);}
    CHECK(!ra_live&&!ra_bytes);printf("PASS %u private copied map observation checks; public boundaries checked.\n",checks);return 0;
}
