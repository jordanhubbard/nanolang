/* I invoke actual shared state transitions and observe their retained facts. */
#include "../src/nanoisa/affine_state.c"
#include <stdio.h>
#include <assert.h>
static uint32_t read32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static bool take(size_t *cursor, size_t length, size_t size, size_t *start) {
    if (length > size - *cursor) return false;
    *start = *cursor; *cursor += length;
    return true;
}
static bool load_model(const unsigned char *bytes,size_t size,NvmModule *output) {
    if (size < 28) { return false; }
    uint32_t layout_size = read32(bytes), own_size = read32(bytes + 4);
    uint32_t strings = read32(bytes + 8), functions = read32(bytes + 12);
    size_t cursor = 28;
    if (strings > (size - cursor) / 4) { return false; }
    NvmModule model = {0};
    model.struct_count = read32(bytes + 16); model.enum_count = read32(bytes + 20);
    model.union_count = read32(bytes + 24);
    char **names = calloc(strings ? strings : 1, sizeof(*names));
    uint32_t *lengths = calloc(strings ? strings : 1, sizeof(*lengths));
    NvmFunctionEntry *entries = NULL; uint8_t **params = NULL;
    bool ok = names && lengths;
    for (uint32_t i = 0; ok && i < strings; ++i) {
        size_t at;
        if (!take(&cursor, 4, size, &at)) { ok = false; break; }
        uint32_t length = read32(bytes + at);
        if (!take(&cursor, length, size, &at)) { ok = false; break; }
        names[i] = malloc((size_t)length + 1);
        if (!names[i]) { ok = false; break; }
        memcpy(names[i], bytes + at, length); names[i][length] = 0; lengths[i] = length;
    }
    if (ok && functions > (size - cursor) / 32) ok = false;
    if (ok) {
        entries = calloc(functions ? functions : 1, sizeof(*entries));
        params = calloc(functions ? functions : 1, sizeof(*params));
        if (!entries || !params) ok = false;
    }
    for (uint32_t i = 0; ok && i < functions; ++i) {
        size_t at;
        if (!take(&cursor, 32, size, &at)) { ok = false; break; }
        const unsigned char *p = bytes + at;
        uint32_t arity = read32(p + 8), locals = read32(p + 12), upvalues = read32(p + 16);
        uint32_t results = read32(p + 20), result_tag = read32(p + 24), types = read32(p + 28);
        if (arity > UINT16_MAX || locals > UINT16_MAX || upvalues > UINT16_MAX || results > UINT8_MAX ||
            result_tag > UINT8_MAX || (types && types != arity) || !take(&cursor, types, size, &at)) { ok = false; break; }
        entries[i].code_offset = read32(p); entries[i].code_length = read32(p + 4);
        entries[i].arity = (uint16_t)arity; entries[i].local_count = (uint16_t)locals;
        entries[i].upvalue_count = (uint16_t)upvalues; entries[i].result_count = (uint8_t)results;
        entries[i].result_tag = (uint8_t)result_tag;
        params[i] = types ? (uint8_t *)(bytes + at) : NULL;
    }
    size_t layout_at = 0, own_at = 0;
    if (ok && (!take(&cursor, layout_size, size, &layout_at) ||
               !take(&cursor, own_size, size, &own_at) || size - cursor > UINT32_MAX)) ok = false;
    if (ok) {
        model.strings = names; model.string_lengths = lengths; model.string_count = strings;
        model.functions = entries; model.function_param_types = params; model.function_count = functions;
        model.layout_data = layout_size ? (uint8_t *)(bytes + layout_at) : NULL; model.layout_size = layout_size;
        model.ownership_data = own_size ? (uint8_t *)(bytes + own_at) : NULL; model.ownership_size = own_size;
        model.code_size = (uint32_t)(size - cursor); model.code = model.code_size ? (uint8_t *)(bytes + cursor) : NULL;
        *output=model;
    }
    if (!ok) {
        if (names)for(uint32_t i=0;i<strings;i++)free(names[i]);
        free(names);free(lengths);free(entries);free(params);
    }
    return ok;
}
static void free_model(NvmModule *m) {
    for(uint32_t i=0;i<m->string_count;i++)free(m->strings[i]);
    free(m->strings);free(m->string_lengths);free(m->functions);free(m->function_param_types);
}

static void number(uint32_t n) {printf("%u,",n);}
static void print_slot(uint8_t tag,uint8_t mode,uint32_t layout) {number(tag);number(mode);number(layout);}
static void snapshot(const NvmAffineState *s,uint32_t identity) {
    if(!s){putchar('N');return;}
    putchar('S');number(identity);number(s->region);number(s->caller_bound);number(s->handler);
    number(s->handler_first);number(s->handler_key);number(s->resume_type.tag);number(s->resume_type.layout);
    number((uint32_t)s->invocation);number((uint32_t)(s->invocation>>32));putchar(':');
    for(uint16_t i=0;i<s->facts->count;i++){putchar(s->live[i]?'1':'0');number(s->variants[i]);}
    putchar(':');
    for(uint32_t i=0;i<s->ref_count;i++) {
        const Reference *r=&s->refs[i];if(!r->live){putchar('-');continue;}
        const NvmReferencePlace *p=&r->place;putchar('R');number(r->region);number(r->parent);
        number((uint32_t)p->invocation);number((uint32_t)(p->invocation>>32));number(p->local);
        number(p->root_layout);number(p->referent_layout);number(p->mode);putchar('[');
        for(uint16_t j=0;j<p->field_count;j++){number(p->fields[j]);}
        putchar(']');
    }
    putchar(':');
    for(uint16_t i=0;i<s->origin_count;i++) {
        const CallerOrigin *p=&s->origins[i];number((uint32_t)p->invocation);number((uint32_t)(p->invocation>>32));number(p->local);number(p->layout);
    }
}

typedef struct {uint32_t op,source,a,b,c,d,count;uint16_t *fields;} Command;
/* I perturb only the oracle's owned path storage, then restore it. This is
 * an ownership/isolation observation, not a new state transition API. */
static void assert_isolated(NvmAffineState *held,const NvmAffineState *source) {
    if(!held || !source)return;
    NvmAffineState *before=nvm_affine_state_clone(held);assert(before);
    for(uint32_t i=0;i<source->ref_count;i++) {
        const Reference *r=&source->refs[i];if(!r->live || !r->place.field_count)continue;
        uint16_t *path=(uint16_t *)r->place.fields,original=path[0];path[0]=(uint16_t)(original+1);
        assert(nvm_affine_state_equal(held,before));path[0]=original;
    }
    nvm_affine_state_free(before);
}
static bool command_read(const uint8_t *bytes,size_t size,size_t *pos,Command *cmd) {
    size_t at;if(!take(pos,28,size,&at))return false;
    const uint8_t *p=bytes+at;
    *cmd=(Command){read32(p),read32(p+4),read32(p+8),read32(p+12),read32(p+16),read32(p+20),read32(p+24),NULL};
    if(cmd->op>51 || cmd->source>=16 || cmd->count>65535 ||
       ((cmd->op==1 || cmd->op==41 || cmd->op==42 || cmd->op==43 || cmd->op==44) && cmd->a>=16) ||
       (cmd->op==0 && (cmd->b>65536 || cmd->c>1)) || !take(pos,cmd->count*4,size,&at))return false;
    const uint32_t locals[]={2,3,4,5,6,7,8,9,10,11,12,17,19,20,21,22,23,31,32,35,45,49,50};
    const uint32_t second[]={7,8,11,12,15,18,22,23,33,34,35,51},third[]={6,34,43},tags[]={24,25,27,29,30,36,37,38};
    for(size_t i=0;i<sizeof(locals)/sizeof(*locals);i++)if(cmd->op==locals[i] && cmd->a>65535)return false;
    for(size_t i=0;i<sizeof(second)/sizeof(*second);i++)if(cmd->op==second[i] && cmd->b>65535)return false;
    for(size_t i=0;i<sizeof(third)/sizeof(*third);i++)if(cmd->op==third[i] && cmd->c>65535)return false;
    for(size_t i=0;i<sizeof(tags)/sizeof(*tags);i++)if(cmd->op==tags[i] && cmd->a>255)return false;
    if((cmd->op==32 && cmd->b>255) || (cmd->op==29 && cmd->c>255))return false;
    cmd->fields=calloc(cmd->count?cmd->count:1,sizeof *cmd->fields);if(!cmd->fields)exit(2);
    for(uint32_t i=0;i<cmd->count;i++){uint32_t v=read32(bytes+at+i*4);if(v>65535)return false;cmd->fields[i]=(uint16_t)v;}
    return true;
}

static void execute(NvmModule *module,Command cmd,NvmAffineState **states,uint32_t *ids,uint32_t *serial) {
    NvmAffineState *s=states[cmd.source],*before=s?nvm_affine_state_clone(s):NULL;
    if(s && !before)exit(2);
    uint32_t op=cmd.op,a=cmd.a,b=cmd.b,c=cmd.c,d=cmd.d,target=cmd.source;
    uint16_t count=(uint16_t)cmd.count,*fields=cmd.fields;
    bool ok=false,changed=false,query=false;
    uint8_t tag=239,mode=239;uint16_t n=51966,variant=51966;
    NvmAffineType type={239,0xecececec},other={0};
    NvmAffineType *values=calloc(65536,sizeof *values);if(!values)exit(2);
    uint8_t report_tag=0,report_mode=0;uint32_t report_layout=0,report_number=0;
    switch(op) {
    case 0: {
        NvmAffineState *created=nvm_affine_state_create(module+c,a,b);++*serial;
        nvm_affine_state_free(states[target]);states[target]=created;ids[target]=created?*serial:0;ok=created!=NULL;break;
    }
    case 1: {
        target=a;NvmAffineState *cloned=nvm_affine_state_clone(s);uint32_t id=ids[cmd.source];
        assert_isolated(cloned,s);
        nvm_affine_state_free(states[target]);states[target]=cloned;ids[target]=cloned?id:0;ok=cloned!=NULL;break;
    }
    case 2:ok=nvm_affine_scalar_define(s,(uint16_t)a);break;
    case 3:ok=nvm_affine_string_define(s,(uint16_t)a);break;
    case 4:ok=nvm_affine_record_define(s,(uint16_t)a,b);break;
    case 5:ok=nvm_affine_array_define(s,(uint16_t)a,b);break;
    case 6:ok=nvm_affine_union_define(s,(uint16_t)a,b,(uint16_t)c);break;
    case 7:ok=nvm_affine_union_refine(s,(uint16_t)a,(uint16_t)b);break;
    case 8:ok=nvm_affine_move(s,(uint16_t)a,(uint16_t)b);break;
    case 9:ok=nvm_affine_pack(s,(uint16_t)a,fields,count);break;
    case 10:ok=nvm_affine_unpack(s,(uint16_t)a,fields,count);break;
    case 11:ok=nvm_affine_union_pack(s,(uint16_t)a,(uint16_t)b,fields,count);break;
    case 12:ok=nvm_affine_union_unpack(s,(uint16_t)a,(uint16_t)b,fields,count);break;
    case 13:ok=nvm_affine_region_begin(s);break;
    case 14:ok=nvm_affine_region_end(s);break;
    case 15:
        ok=nvm_affine_borrow(s,a,(uint16_t)b,fields,count,(NvmReferenceMode)c);
        if(ok && count) {
            NvmAffineState *copy=nvm_affine_state_clone(s);assert(copy);uint16_t original=fields[0];fields[0]=(uint16_t)(original+1);
            assert(nvm_affine_state_equal(s,copy));fields[0]=original;nvm_affine_state_free(copy);
        }
        break;
    case 16:ok=nvm_affine_reborrow(s,a,b,(NvmReferenceMode)c);break;
    case 17:ok=nvm_affine_owner_access(s,(uint16_t)a,fields,count,b!=0);break;
    case 18:ok=nvm_affine_reference_access(s,a,(uint16_t)b,c!=0);break;
    case 19:ok=nvm_affine_can_exit(s,(uint16_t)a);break;
    case 20:
        query=true;ok=nvm_affine_local_info(s,(uint16_t)a,&tag,&mode);
        if(ok){report_tag=tag;report_mode=mode;report_layout=s->facts->locals[a].layout;}break;
    case 21:
        query=true;ok=nvm_affine_local_type(s,(uint16_t)a,&type);
        if(ok){report_tag=type.tag;report_layout=type.layout;}break;
    case 22:
        query=true;ok=nvm_affine_scalar_field(s,(uint16_t)a,(uint16_t)b,&tag);
        if(ok){report_tag=tag;report_layout=NVM_V2_NO_INDEX;}break;
    case 23:
        query=true;ok=nvm_affine_string_field(s,(uint16_t)a,(uint16_t)b,&tag);
        if(ok){report_tag=tag;report_layout=NVM_V2_NO_INDEX;}break;
    case 24:ok=nvm_affine_can_exit_scalar(s,(uint8_t)a);break;
    case 25:ok=nvm_affine_can_exit_type(s,(NvmAffineType){(uint8_t)a,b});break;
    case 26:ok=nvm_affine_can_suspend(s);break;
    case 27:ok=nvm_affine_can_resume(s,(NvmAffineType){(uint8_t)a,b});break;
    case 28:
        query=true;ok=nvm_affine_union_variant(s,(uint16_t)a,&variant);
        if(ok){report_tag=s->facts->locals[a].tag;report_layout=s->facts->locals[a].layout;report_number=variant;}break;
    case 29:ok=nvm_affine_types_equal(s,(NvmAffineType){(uint8_t)a,b},(NvmAffineType){(uint8_t)c,d});break;
    case 30:ok=nvm_affine_type_is_owned(s,(NvmAffineType){(uint8_t)a,b});break;
    case 31:
        ok=nvm_affine_take_local(s,(uint16_t)a,&type);query=true;
        if(ok){report_tag=type.tag;report_layout=type.layout;}break;
    case 32:ok=nvm_affine_put_local(s,(uint16_t)a,(NvmAffineType){(uint8_t)b,c});break;
    case 33:
        query=true;ok=nvm_affine_record_fields(s,a,c?values:NULL,(uint16_t)b,&n);
        if(ok){report_tag=TAG_STRUCT;report_layout=a;report_number=n;}break;
    case 34:
        query=true;ok=nvm_affine_union_fields(s,a,(uint16_t)b,d?values:NULL,(uint16_t)c,&n);
        if(ok){report_tag=TAG_UNION;report_layout=a;report_number=n;}break;
    case 35:
        query=true;ok=nvm_affine_owned_local_fields(s,(uint16_t)a,c?values:NULL,(uint16_t)b,&n);
        if(ok){report_tag=s->facts->locals[a].tag;report_layout=s->facts->locals[a].layout;report_number=n;}break;
    case 36:
        query=true;ok=nvm_affine_array_element(s,(NvmAffineType){(uint8_t)a,b},&type);
        if(ok){report_tag=type.tag;report_layout=type.layout;}break;
    case 37:
        query=true;ok=nvm_affine_array_type(s,(NvmAffineType){(uint8_t)a,b},&type);
        if(ok){report_tag=type.tag;report_layout=type.layout;}break;
    case 38:
        query=true;ok=nvm_affine_array_tag_type(s,(uint8_t)a,&type);
        if(ok){report_tag=type.tag;report_layout=type.layout;}break;
    case 39:ok=nvm_affine_has_complete_unions(s);break;
    case 40:ok=nvm_affine_has_array_types(s);break;
    case 41:ok=nvm_affine_state_equal(s,states[a]);break;
    case 42:ok=nvm_affine_state_meet_initialization(s,states[a],&changed);break;
    case 43: {
        target=a;NvmAffineState *arm=nvm_affine_handler_enter(s,b,(uint16_t)c);uint32_t id=ids[cmd.source];
        nvm_affine_state_free(states[target]);states[target]=arm;ids[target]=arm?id:0;ok=arm!=NULL;break;
    }
    case 44:ok=nvm_affine_bind_caller(s,states[a],b);if(ok)assert_isolated(s,states[a]);break;
    case 45:case 46: {
        query=true;NvmReferenceMode result_mode=(NvmReferenceMode)239;
        ok=op==45?nvm_affine_parameter_at(s,(uint16_t)a,&type,&result_mode):nvm_affine_parameter_type(s,&type,&result_mode);
        if(ok){report_tag=type.tag;report_mode=(uint8_t)result_mode;report_layout=type.layout;}else assert(result_mode==(NvmReferenceMode)239);break;
    }
    case 47:
        query=true;ok=nvm_affine_owned_parameter_type(s,&type);
        if(ok){report_tag=type.tag;report_layout=type.layout;}break;
    case 48:
        query=true;ok=nvm_affine_value_result(s,&type,&n);
        if(ok){report_tag=type.tag;report_layout=type.layout;report_number=n;}break;
    case 49:case 50:
        query=true;ok=op==49?nvm_affine_value_parameters(s,b?values:NULL,(uint16_t)a,&n):nvm_affine_consuming_parameters(s,b?values:NULL,(uint16_t)a,&n);
        if(ok){report_number=n;}
        break;
    case 51:
        query=true;ok=nvm_affine_reference_field(s,a,(uint16_t)b,c!=0,&tag);
        if(ok){report_tag=tag;report_layout=NVM_V2_NO_INDEX;}break;
    }
    if(!ok) {
        if(before && op!=0 && op!=1 && op!=43)assert(nvm_affine_state_equal(s,before));
        assert(tag==239 && mode==239 && type.tag==239 && type.layout==0xecececec && n==51966 && variant==51966 && !changed);
        if(op==33 || op==34 || op==35 || op==49 || op==50)
            for(size_t i=0;i<65536;i++)assert(values[i].tag==0 && values[i].layout==0);
    }
    printf("%u%u",ok?1u:0u,changed?1u:0u);
    if(query && ok) {
        putchar(':');print_slot(report_tag,report_mode,report_layout);number(report_number);
        if(op==33 || op==34 || op==35 || op==49 || op==50)
            for(uint32_t i=0;i<report_number;i++)print_slot(values[i].tag,0,values[i].layout);
    }
    putchar('@');snapshot(states[target],ids[target]);
    nvm_affine_state_free(before);free(values);(void)other;
}

static void program(const uint8_t *bytes,size_t size) {
    if(size<8){puts("D");return;}
    uint32_t model_size=read32(bytes),count=read32(bytes+4);size_t pos=8,at;
    if(count>65536 || !take(&pos,model_size,size,&at)){puts("D");return;}
    NvmModule modules[2];memset(modules,0,sizeof modules);
    if(!load_model(bytes+at,model_size,&modules[0])){puts("D");return;}
    modules[1]=modules[0];Command *commands=calloc(count?count:1,sizeof *commands);if(!commands)exit(2);
    bool valid=true;
    for(uint32_t i=0;i<count;i++)if(!command_read(bytes,size,&pos,&commands[i])){valid=false;break;}
    if(pos!=size)valid=false;
    if(!valid)puts("D");
    else {
        NvmAffineState *states[16]={0};uint32_t identities[16]={0},serial=0;
        for(uint32_t i=0;i<count;i++){if(i)putchar('|');execute(modules,commands[i],states,identities,&serial);}
        putchar('\n');for(unsigned i=0;i<16;i++)nvm_affine_state_free(states[i]);
    }
    for(uint32_t i=0;i<count;i++)free(commands[i].fields);
    free(commands);free_model(&modules[0]);
}
int main(int argc,char **argv) {
    for(int arg=1;arg<argc;arg++) {
        FILE *file=fopen(argv[arg],"rb");if(!file || fseek(file,0,SEEK_END))return 2;
        long size=ftell(file);if(size<0 || fseek(file,0,SEEK_SET))return 2;
        uint8_t *bytes=malloc(size?(size_t)size:1);if(!bytes)return 2;
        if(fread(bytes,1,(size_t)size,file)!=(size_t)size || fclose(file))return 2;
        program(bytes,(size_t)size);free(bytes);
    }
    return 0;
}
