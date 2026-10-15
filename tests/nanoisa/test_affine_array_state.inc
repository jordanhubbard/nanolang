/* I exercise exact array state independently of bytecode admission. */
static NvmModule *array_state_fixture(void) {
    AsmResult error;
    NvmModule *m=asm_assemble_unverified(".string pass \"Array.pass\"\n.string tick \"Array.tick\"\n.entry 0\n.function main 0 11 0 array 1\nPUSH_I64 0\nRET\n.end\n",&error);
    CHECK(m);
    NvmV2LayoutField field={TAG_INT,NVM_V2_NO_INDEX,NVM_V2_NO_INDEX};
    NvmV2Layout records[]={{NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field},
                          {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field},
                          {NVM_V2_LAYOUT_STRUCT,1,NVM_V2_NO_INDEX,&field}};
    NvmV2Layouts layouts={records,3};m->struct_count=3;
    CHECK(nvm_retain_layouts(m,&layouts)==NVM_V2_OK);
    uint8_t *p=calloc(264,1);CHECK(p);m->ownership_data=p;m->ownership_size=264;
    word(p,3);word(p+4,3);p[8]=p[9]=3;p[10]=1;word(p+12,1);p[16]=11;
    slot(p+20,TAG_ARRAY,0,0);
    const unsigned refs[]={0,0,1,2,3,4,5,6,7,8,9};
    for(unsigned i=0;i<11;i++)slot(p+28+8*i,TAG_ARRAY,0,refs[i]);
    word(p+116,4);word(p+124,1);p[128]=NVM_OWNERSHIP_EXTENSION_EFFECTS;p[130]=2;
    word(p+132,128);word(p+136,10);
    slot(p+140,TAG_STRUCT,0,0);slot(p+148,TAG_STRUCT,0,0);slot(p+156,TAG_STRUCT,0,1);
    slot(p+164,TAG_INT,0,NVM_V2_NO_INDEX);slot(p+172,TAG_INT,0,NVM_V2_NO_INDEX);
    slot(p+180,TAG_ARRAY,0,0);slot(p+188,TAG_ARRAY,0,2);
    slot(p+196,TAG_ARRAY,0,3);slot(p+204,TAG_ARRAY,0,4);
    slot(p+212,TAG_STRUCT,0,2);
    word(p+220,2);word(p+224,0);p[228]=1;slot(p+232,TAG_ARRAY,0,0);slot(p+240,TAG_ARRAY,0,0);
    word(p+248,1);slot(p+256,TAG_INT,0,NVM_V2_NO_INDEX);
    bool needed=false;CHECK(nvm_ownership_contracts_validate(m,&needed)==NVM_V2_OK && needed);
    return m;
}
static void array_states(void) {
    NvmModule *m=array_state_fixture();NvmAffineState *s=nvm_affine_state_create(m,0,2);CHECK(s);
    CHECK(nvm_affine_has_array_types(s));
    NvmAffineType owner={TAG_ARRAY,0},alias={TAG_ARRAY,1},other={TAG_ARRAY,2},nested={TAG_ARRAY,5};
    CHECK(nvm_affine_types_equal(s,owner,alias));CHECK(!nvm_affine_types_equal(s,owner,other));
    CHECK(nvm_affine_types_equal(s,(NvmAffineType){TAG_ARRAY,7},(NvmAffineType){TAG_ARRAY,8}));
    CHECK(!nvm_affine_types_equal(s,owner,nested));
    CHECK(!nvm_affine_types_equal(s,(NvmAffineType){TAG_ARRAY,99},(NvmAffineType){TAG_ARRAY,99}));
    CHECK(nvm_affine_type_is_owned(s,owner));CHECK(nvm_affine_type_is_owned(s,nested));
    CHECK(!nvm_affine_type_is_owned(s,(NvmAffineType){TAG_ARRAY,7}));
    CHECK(!nvm_affine_type_is_owned(s,(NvmAffineType){TAG_ARRAY,99}));
    NO_CHANGE(s,nvm_affine_array_define(s,0,0));
    CHECK(nvm_affine_put_local(s,0,owner));CHECK(!nvm_affine_can_suspend(s));
    NO_CHANGE(s,nvm_affine_put_local(s,0,owner));NO_CHANGE(s,nvm_affine_move(s,0,3));
    CHECK(nvm_affine_move(s,0,2));NvmAffineType got={TAG_VOID,99};
    CHECK(nvm_affine_take_local(s,2,&got));CHECK(got.tag==TAG_ARRAY && got.layout==1);
    CHECK(nvm_affine_can_suspend(s));NO_CHANGE(s,nvm_affine_take_local(s,2,&got));
    CHECK(nvm_affine_put_local(s,1,got));CHECK(nvm_affine_can_exit(s,1));
    CHECK(nvm_affine_take_local(s,1,&got));CHECK(nvm_affine_can_exit_type(s,alias));
    CHECK(!nvm_affine_can_exit_type(s,other));
    CHECK(nvm_affine_array_define(s,4,4));CHECK(nvm_affine_array_define(s,8,8));
    CHECK(!nvm_affine_type_is_owned(s,(NvmAffineType){TAG_ARRAY,9}));
    CHECK(nvm_affine_array_define(s,10,9));
    CHECK(nvm_affine_can_suspend(s));NO_CHANGE(s,nvm_affine_array_define(s,4,7));
    NvmAffineState *before=nvm_affine_state_clone(s);CHECK(before);
    CHECK(nvm_affine_array_define(s,5,3));bool changed=false;
    CHECK(nvm_affine_state_meet_initialization(s,before,&changed) && changed);
    uint8_t tag=99,mode=99;CHECK(!nvm_affine_local_info(s,5,&tag,&mode));
    CHECK(tag==99 && mode==99);nvm_affine_state_free(before);
    CHECK(nvm_affine_put_local(s,6,nested));
    before=nvm_affine_state_clone(s);CHECK(before);CHECK(nvm_affine_take_local(s,6,&got));
    NvmAffineState *after=nvm_affine_state_clone(s);CHECK(after);
    CHECK(!nvm_affine_state_meet_initialization(s,before,&changed));CHECK(nvm_affine_state_equal(s,after));
    nvm_affine_state_free(before);nvm_affine_state_free(after);
    CHECK(nvm_affine_put_local(s,6,nested));
    uint16_t field=0,count=99;NvmAffineType field_type={TAG_VOID,99};
    CHECK(!nvm_affine_scalar_field(s,6,0,&tag));CHECK(!nvm_affine_string_field(s,6,0,&tag));
    CHECK(!nvm_affine_owned_local_fields(s,6,&field_type,1,&count));CHECK(count==99);
    NO_CHANGE(s,nvm_affine_pack(s,6,&field,1));NO_CHANGE(s,nvm_affine_unpack(s,6,&field,1));
    CHECK(!nvm_affine_owner_access(s,6,&field,1,false));
    CHECK(nvm_affine_region_begin(s));NO_CHANGE(s,nvm_affine_borrow(s,0,6,NULL,0,NVM_REFERENCE_SHARED));
    CHECK(nvm_affine_region_end(s));CHECK(nvm_affine_take_local(s,6,&got));
    CHECK(nvm_affine_put_local(s,0,owner));before=nvm_affine_state_clone(s);CHECK(before);
    NvmAffineState *arm=nvm_affine_handler_enter(s,0,1);CHECK(arm);
    CHECK(nvm_affine_state_equal(s,before));NO_CHANGE(arm,nvm_affine_take_local(arm,0,&got));
    NO_CHANGE(arm,nvm_affine_put_local(arm,0,owner));
    CHECK(!nvm_affine_can_resume(arm,owner));CHECK(nvm_affine_take_local(arm,1,&got));
    CHECK(nvm_affine_can_resume(arm,alias));CHECK(!nvm_affine_can_resume(arm,other));
    CHECK(nvm_affine_can_exit_type(arm,alias));nvm_affine_state_free(arm);nvm_affine_state_free(before);
    arm=nvm_affine_handler_enter(s,1,10);CHECK(arm);
    CHECK(nvm_affine_local_info(arm,4,&tag,&mode) && tag==TAG_ARRAY);
    CHECK(nvm_affine_array_define(arm,4,4));
    CHECK(nvm_affine_can_resume(arm,(NvmAffineType){TAG_INT,NVM_V2_NO_INDEX}));
    nvm_affine_state_free(arm);
#ifdef AFFINE_ALLOCATION_TEST
    for(unsigned failure=1;;failure++) {
        allocation_attempts=0;fail_at=failure;NvmAffineState *trial=nvm_affine_state_create(m,0,2);fail_at=0;
        if(trial){nvm_affine_state_free(trial);break;}CHECK(failure<32);
        trial=nvm_affine_state_create(m,0,2);CHECK(trial);nvm_affine_state_free(trial);
    }
    before=nvm_affine_state_clone(s);CHECK(before);
    for(unsigned failure=1;;failure++) {
        allocation_attempts=0;fail_at=failure;NvmAffineState *trial=nvm_affine_handler_enter(s,0,1);fail_at=0;
        CHECK(nvm_affine_state_equal(s,before));
        if(trial){nvm_affine_state_free(trial);break;}CHECK(failure<32);
        trial=nvm_affine_handler_enter(s,0,1);CHECK(trial);nvm_affine_state_free(trial);
    }
    nvm_affine_state_free(before);
#endif
    nvm_affine_state_free(s);nvm_module_free(m);
}
