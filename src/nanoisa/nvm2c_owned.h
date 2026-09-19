/* I lower the already verified standalone ownership contract to direct C
 * operations and constant-index temporaries. I emit no bytecode interpreter. */
#include "mixed_samples_internal.h"
#include "managed_native_source.h"
#include "owned_array_authority.h"
#include "owned_array_runtime_private.h"
static bool owned_emission_local(const NvmAffineState *state,const NvmMixedSamplesPlan *mixed,
                                  const NvmOwnedArrayPlan *owner_arrays,uint32_t function,uint16_t local,NvmAffineType *out) {
    if(owner_arrays) {
        NvmOwnerDeclaration d;
        if(!nvm_owned_array_plan_local(owner_arrays,function,local,&d) || !d.owner)return false;
        *out=(NvmAffineType){d.tag,d.global_layout};return true;
    }
    if(!mixed)return nvm_affine_local_type(state,local,out);
    NvmMixedDeclaration d;
    if(!nvm_mixed_samples_local(mixed,function,local,&d) || d.category!=NVM_MIXED_VALUE_OWNER)return false;
    *out=(NvmAffineType){d.tag,d.global_layout};return true;
}
static char *emit_owned_function(const NvmModule *mod,uint32_t function,
                                  const NvmMixedSamplesPlan *mixed,const NvmOwnedArrayPlan *owner_arrays,char *err,size_t err_len) {
    const bool shared=mixed || owner_arrays;
    Nvm2cBuf b={0}; b.err=err; b.err_len=err_len;
    VmDecodedFunction code={0}; char decode_error[VM_DECODE_ERROR_SIZE];
    NvmAffineState *state=NULL; NvmV2Layouts layouts={0};
    int *depth=NULL; uint32_t *queue=NULL;
    const NvmFunctionEntry *fn=&mod->functions[function];
    if (!vm_decode_function(mod,function,&code,decode_error) ||
        nvm_v2_layouts_decode(mod->layout_data,mod->layout_size,&layouts)!=NVM_V2_OK ||
        (!shared && !(state=nvm_affine_state_create(mod,function,fn->local_count)))) goto fail;
    NvmAffineType parameters[NVM_AFFINE_MAX_PARAMETERS];uint16_t parameter_count=0;
    bool value_graph=shared || nvm_affine_value_call_graph(mod);
    bool consuming=function && value_graph;
    NvmAffineType result_type;uint16_t result_fields=0;
    if(owner_arrays) {
        NvmOwnerSignature signature;
        if(!nvm_owned_array_plan_signature(owner_arrays,function,&signature))goto fail;
        result_type=(NvmAffineType){signature.result.tag,signature.result.global_layout};
        result_fields=signature.result_fields;parameter_count=signature.parameter_count;
        for(uint16_t n=0;n<parameter_count;n++)parameters[n]=(NvmAffineType){signature.parameters[n].tag,signature.parameters[n].global_layout};
    } else if(mixed) {
        NvmMixedSignature signature;
        if(!nvm_mixed_samples_signature(mixed,function,&signature))goto fail;
        result_type=(NvmAffineType){signature.result.tag,signature.result.global_layout};
        result_fields=signature.result_fields;parameter_count=signature.parameter_count;
        for(uint16_t n=0;n<parameter_count;n++)parameters[n]=(NvmAffineType){signature.parameters[n].tag,signature.parameters[n].global_layout};
    } else {
        if (!nvm_affine_value_result(state,&result_type,&result_fields)) goto fail;
        if (consuming && (!nvm_affine_value_parameters(state,parameters,NVM_AFFINE_MAX_PARAMETERS,&parameter_count) ||
                          parameter_count!=fn->arity)) goto fail;
    }
    depth=malloc(code.instruction_count*sizeof(*depth));
    queue=malloc(code.instruction_count*sizeof(*queue));
    if (!depth || !queue) goto fail;
    for (uint32_t i=0;i<code.instruction_count;i++) depth[i]=-1;
    depth[0]=0; queue[0]=0; uint32_t head=0,tail=1;
    while (head<tail) {
        uint32_t i=queue[head++]; const VmDecodedInstruction *d=&code.instructions[i];
        const DecodedInstruction *in=&d->instruction; uint8_t op=in->opcode;
        const InstructionInfo *info=isa_get_info(op);
        int pop=info->pop_count,push=info->push_count;
        if (op==OP_OWN_PACK) {pop=layouts.items[in->operands[0].u32].field_count;push=1;}
        if (op==OP_OWN_UNPACK_LOCAL) {
            NvmAffineType type;
            if (!owned_emission_local(state,mixed,owner_arrays,function,in->operands[0].u16,&type)) goto fail;
            pop=0;push=layouts.items[type.layout].field_count;
        }
        if(mixed && op==OP_AGG_PACK){pop=in->operands[3].u16;push=1;}
        if(shared && op==OP_ARR_LITERAL){pop=in->operands[1].u16;push=1;}
        if (op==OP_CALL) {pop=mod->functions[in->operands[0].u32].arity;push=mod->functions[in->operands[0].u32].result_count;}
        if (op==OP_RET) continue;
        if (pop<0 || push<0 || depth[i]<pop || depth[i]-pop+push>(int)NVM_AFFINE_MAX_STACK) goto fail;
        uint32_t targets[2],count=0;
        if (op==OP_JMP || op==OP_JMP_TRUE || op==OP_JMP_FALSE) {
            const VmDecodedInstruction *target=vm_decoded_function_at(&code,d->resolved_target-fn->code_offset);
            if (!target) goto fail;
            targets[count++]=(uint32_t)(target-code.instructions);
        }
        if (op!=OP_JMP) targets[count++]=i+1;
        for (uint32_t j=0;j<count;j++) {
            uint32_t target=targets[j];int value=depth[i]-pop+push;
            if (target>=code.instruction_count) goto fail;
            if (depth[target]<0) {depth[target]=value;queue[tail++]=target;}
            else if (depth[target]!=value) goto fail;
        }
    }
    bool float_arithmetic=module_has_opcode(mod,OP_F64_ADD) || module_has_opcode(mod,OP_F64_SUB) ||
        module_has_opcode(mod,OP_F64_MUL) || module_has_opcode(mod,OP_F64_DIV);
    if(!function && shared)nvm2c_puts(&b,nms_native_source);
    if (!function && float_arithmetic) nvm2c_puts(&b,nl_binary64_arithmetic_source);
    if (!function) nvm2c_puts(&b,
        "/* I transfer unique owners; field observations retain only their shell. */\n"
        "#include <stdint.h>\n#include <stdlib.h>\n#include <stddef.h>\n#include <limits.h>\n#include <stdio.h>\n#include <string.h>\n"
        "#ifndef NOWN_ALLOC\n#define NOWN_ALLOC calloc\n#endif\n"
        "#ifndef NOWN_FREE\n#define NOWN_FREE free\n#endif\n"
        "typedef struct nown_record nown_record;\n"
        "typedef struct { size_t refs,length; unsigned char bytes[]; } nown_string;\n"
        );
    if(!function && shared)nvm2c_puts(&b,
        "typedef struct { int64_t scalar; nown_record *record; nown_string *string; double floating; uint8_t tag,category; NmsRuntime *managed; NmsHandle handle; } nown_value;\n");
    else if(!function)nvm2c_puts(&b,
        "typedef struct { int64_t scalar; nown_record *record; nown_string *string; double floating; uint8_t tag; } nown_value;\n");
    if(!function)nvm2c_puts(&b,
        "static inline double nown_float_bits(uint64_t bits) { double value; memcpy(&value,&bits,sizeof(value)); return value; }\n"
        "struct nown_record { size_t refs, count; uint32_t layout; nown_value fields[]; };\n"
        "typedef struct { unsigned root,region,exclusive,parent,depth,origin; uint64_t generation; uint16_t fields[32]; } nown_reference;\n"
        "static nown_record *nown_referent(nown_value *const origins[2],const uint64_t generations[2],const nown_reference *ref) {\n"
        " if(ref->origin>1 || !origins[ref->origin] || ref->generation!=generations[ref->origin])return NULL;\n"
        " nown_record *record=origins[ref->origin][ref->root].record;\n"
        " for(unsigned i=0;i<ref->depth;i++) record=record->fields[ref->fields[i]].record;\n"
        " return record;\n}\n"
        "static void nown_release(nown_value v) {\n");
    if(!function && shared)nvm2c_puts(&b,
        " if(v.category==2){(void)nms_release(v.managed,v.handle);return;}\n");
    if(!function)nvm2c_puts(&b,
        " if (v.string && --v.string->refs==0) NOWN_FREE(v.string);\n"
        " if (v.record && --v.record->refs==0) {\n"
        "  for(size_t i=0;i<v.record->count;i++) nown_release(v.record->fields[i]);\n"
        "  NOWN_FREE(v.record);\n }\n}\n"
        "static int nown_retain(nown_value v) {\n");
    if(!function && shared)nvm2c_puts(&b,
        " if(v.category==2)return nms_retain(v.managed,v.handle)==NMS_OK;\n");
    if(!function)nvm2c_puts(&b,
        " if((v.string && v.string->refs==SIZE_MAX) || (v.record && v.record->refs==SIZE_MAX)) return 0;\n"
        " if(v.string) ++v.string->refs;\n if(v.record) ++v.record->refs;\n return 1;\n}\n"
        "static nown_string *nown_string_new(const unsigned char *bytes,size_t length) {\n"
        " if(length>SIZE_MAX-sizeof(nown_string)-1) return NULL;\n"
        " nown_string *s=NOWN_ALLOC(1,sizeof(*s)+length+1); if(!s) return NULL;\n"
        " s->refs=1; s->length=length; if(length) memcpy(s->bytes,bytes,length); s->bytes[length]=0; return s;\n}\n"
        "static int nown_string_equal(const nown_string *a,const nown_string *b) {\n"
        " if(a==b) return 1;\n if(!a || !b) return 0;\n"
        " return a->length==b->length && !memcmp(a->bytes,b->bytes,a->length);\n}\n"
        "/* I return status separately so allocation failure still cleans every root. */\n");
    if(!function && shared) {
        nvm2c_puts(&b,
            "static int64_t nown_i64_bits(uint64_t bits){return bits<=INT64_MAX?(int64_t)bits:-1-(int64_t)(UINT64_MAX-bits);}\n"
            "static int nown_status(NmsStatus s){return s==NMS_OK?0:s==NMS_MEMORY?1:s==NMS_ASSERT?2:s==NMS_BOUNDS?4:3;}\n"
            "static int nown_to_managed(nown_value v,NmsValue *out){\n"
            " if(v.category==2){*out=(NmsValue){v.handle,v.tag};return 1;}\n"
            " if(v.record || v.string || v.tag>4)return 0;\n"
            " uint64_t bits=(uint64_t)v.scalar;if(v.tag==3)memcpy(&bits,&v.floating,8);*out=(NmsValue){bits,v.tag};return 1;\n}\n"
            "static int nown_from_managed(NmsRuntime *runtime,NmsValue v,nown_value *out){\n"
            " nown_value result={0};result.tag=(uint8_t)v.tag;\n"
            " if(v.tag==7 || v.tag==8){result.category=2;result.managed=runtime;result.handle=v.payload;}\n"
            " else if(v.tag==3)memcpy(&result.floating,&v.payload,8);\n"
            " else if(v.tag<=4)result.scalar=v.payload<=INT64_MAX?(int64_t)v.payload:-1-(int64_t)(UINT64_MAX-v.payload);\n"
            " else return 0;\n *out=result;return 1;\n}\n");
        if(mixed) {
        const NvmMixedSamplesProof *proof=nvm_mixed_samples_obligations(mixed);
        nvm2c_printf(&b,"static const NmsRecordDescriptor nown_descriptors[%u]={",proof->managed_count);
        for(uint32_t i=0;i<proof->managed_count;i++) {
            uint32_t global=proof->managed_to_global[i];
            nvm2c_printf(&b,"%s{%u,%u}",i?",":"",global,layouts.items[global].field_count);
        }
        nvm2c_puts(&b,"};\n");
        }
    }
    if (!function) {
        if (value_graph) for (uint32_t f=1;f<mod->function_count;f++)
            nvm2c_printf(&b,"static int nown_function_%u(nown_value *argument,uint64_t *next_generation,uint64_t generation,nown_value *result%s);\n",f,shared?",NmsRuntime *managed":"");
        for (uint32_t f=1;f<mod->function_count;f++) {
            char *helper=emit_owned_function(mod,f,mixed,owner_arrays,err,err_len);
            if(!helper) goto fail;
            nvm2c_puts(&b,helper);free(helper);
        }
    }
    if (consuming)
        nvm2c_printf(&b,"static int nown_function_%u(nown_value *argument,uint64_t *next_generation,uint64_t generation,nown_value *result%s) {\n",function,shared?",NmsRuntime *managed":"");
    else nvm2c_puts(&b,function?
        "static int nown_helper(nown_value *origin,const nown_reference *borrowed,uint64_t caller_generation,uint64_t generation,int64_t *result) {\n":
        "int nvm_owned_entry(int64_t *result) {\n");
    nvm2c_puts(&b,
        " nown_value t[256]={{0}}, l[256]={{0}}, a={0}, c={0}, pending={0};\n"
        " nown_reference refs[256]={{0}}; unsigned region=0;\n"
        " int status=0; (void)result; (void)a; (void)c; (void)nown_retain; (void)nown_string_new; (void)nown_string_equal; (void)refs; (void)region; (void)nown_referent;\n");
    if(shared) {
        nvm2c_puts(&b," (void)nown_to_managed;(void)nown_from_managed;(void)nown_status;(void)nown_i64_bits;\n");
        if(!function && owner_arrays) {
            nvm2c_puts(&b," NmsRuntime managed_storage; NmsRuntime *managed=&managed_storage; nms_init(managed,NULL,0);\n status=nown_status(nms_bind_records(managed,NULL,0));if(status)goto cleanup;\n status=nown_status(nms_begin(managed));if(status)goto cleanup;\n");
        } else if(!function) {
            const NvmMixedSamplesProof *proof=nvm_mixed_samples_obligations(mixed);
            nvm2c_printf(&b," NmsRuntime managed_storage; NmsRuntime *managed=&managed_storage; nms_init(managed,NULL,0);\n status=nown_status(nms_bind_records(managed,nown_descriptors,%u));if(status)goto cleanup;\n status=nown_status(nms_begin(managed));if(status)goto cleanup;\n",proof->managed_count);
        } else {
            nvm2c_puts(&b," if(!managed || !managed->active){status=3;goto cleanup;}\n");
            for(uint16_t p=0;p<parameter_count;p++) {
                nvm2c_printf(&b," if(argument[%u].tag!=%u",p,parameters[p].tag);
                if(parameters[p].tag==TAG_STRUCT)nvm2c_printf(&b," || argument[%u].category!=1 || !argument[%u].record || argument[%u].record->layout!=%u",p,p,p,parameters[p].layout);
                else nvm2c_printf(&b," || argument[%u].category",p);
                nvm2c_puts(&b,"){status=3;goto cleanup;}\n");
            }
        }
    }
    if(function) {
        nvm2c_puts(&b,consuming?
            " nown_value *origins[2]={NULL,l}; uint64_t generations[2]={0,generation}; (void)argument; (void)next_generation;\n":
            " nown_value *origins[2]={origin,l}; uint64_t generations[2]={caller_generation,generation};\n");
        if(consuming) for(uint16_t p=0;p<parameter_count;p++)
            nvm2c_printf(&b," l[%u]=argument[%u]; argument[%u]=(nown_value){0};\n",p,p,p);
        for(uint16_t p=0;!consuming && p<fn->arity;p++) {
            NvmAffineType param;NvmReferenceMode mode;
            if(!nvm_affine_parameter_at(state,p,&param,&mode)) goto fail;
            nvm2c_printf(&b," refs[%u]=borrowed[%u]; refs[%u].region=0; refs[%u].parent=UINT16_MAX; refs[%u].exclusive=%u;\n",p,p,p,p,p,mode==NVM_REFERENCE_EXCLUSIVE);
        }
    } else nvm2c_puts(&b," nown_value *origins[2]={l,NULL}; uint64_t generations[2]={1,0},next_generation=1; (void)next_generation;\n");
    if (!function && value_graph) for (uint32_t f=1;f<mod->function_count;f++)
        nvm2c_printf(&b," (void)nown_function_%u;\n",f);
    nvm2c_puts(&b," (void)nown_float_bits;\n");
    if(float_arithmetic) nvm2c_puts(&b," (void)nano_rt_f64_add; (void)nano_rt_f64_sub; (void)nano_rt_f64_mul; (void)nano_rt_f64_div;\n");
    nvm2c_puts(&b," (void)origins; (void)generations; goto L0;\n");
    for (uint32_t i=0;i<code.instruction_count;i++) {
        if (depth[i]<0) continue;
        const VmDecodedInstruction *d=&code.instructions[i];
        const DecodedInstruction *in=&d->instruction;uint8_t op=in->opcode;int n=depth[i];
        unsigned local=in->operands[0].u16;
        if(!shared && (op==OP_PUSH_VOID || op==OP_ROT3 ||
           (op>=OP_I64_ADD && op<=OP_I64_NEG) || (op>=OP_I64_EQ && op<=OP_I64_GE_S) ||
           op==OP_BOOL_AND || op==OP_BOOL_OR || op==OP_BOOL_NOT))goto fail;
        nvm2c_printf(&b,"L%u:;\n",d->byte_offset);
        switch(op) {
        case OP_CALL: {
            uint32_t target=in->operands[0].u32;
            int first=n-mod->functions[target].arity;
            const char *sequence=function?"*next_generation":"next_generation";
            nvm2c_printf(&b," if(%s==UINT64_MAX){status=3;goto cleanup;}\n status=nown_function_%u(&t[%d],%s,++%s,",
                sequence,target,first,function?"next_generation":"&next_generation",sequence);
            if (mod->functions[target].result_count) nvm2c_printf(&b,"&t[%d]",first);
            else nvm2c_puts(&b,"NULL");
            if(shared)nvm2c_puts(&b,",managed");
            nvm2c_puts(&b,"); if(status)goto cleanup;\n");
            break;
        }
        case OP_CALL_REF:
            nvm2c_printf(&b," if(next_generation==UINT64_MAX){status=3;goto cleanup;}\n status=nown_helper(l,&refs[%u],generations[0],++next_generation,&t[%d].scalar); if(status)goto cleanup;\n",in->operands[1].u16,n);
            nvm2c_printf(&b," t[%d].tag=%u;\n",n,mod->functions[in->operands[0].u32].result_tag);break;
        case OP_REGION_BEGIN:nvm2c_puts(&b," ++region;\n");break;
        case OP_REGION_END:
            nvm2c_puts(&b," for(unsigned r=0;r<256;r++) if(refs[r].region==region) refs[r].region=0;\n --region;\n");break;
        case OP_BORROW_PATH_SHARED: case OP_BORROW_PATH_EXCLUSIVE:
        case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE: {
            uint16_t fields[NVM_OWNERSHIP_MAX_PATH_DEPTH],count=0;
            bool exclusive=op==OP_BORROW_LOCAL_EXCLUSIVE || op==OP_BORROW_PATH_EXCLUSIVE;
            if ((op==OP_BORROW_PATH_SHARED || op==OP_BORROW_PATH_EXCLUSIVE) &&
                nvm_ownership_path(mod,in->operands[2].u32,fields,NVM_OWNERSHIP_MAX_PATH_DEPTH,&count)!=NVM_V2_OK) goto fail;
            nvm2c_printf(&b," refs[%u]=(nown_reference){.root=%u,.region=region,.exclusive=%u,.parent=65535,.depth=%u};\n",
                local,in->operands[1].u16,exclusive,count);
            nvm2c_printf(&b," refs[%u].origin=%u; refs[%u].generation=generations[%u];\n",local,function?1u:0u,local,function?1u:0u);
            for (uint16_t j=0;j<count;j++) nvm2c_printf(&b," refs[%u].fields[%u]=%u;\n",local,j,fields[j]);
            break;
        }
        case OP_REBORROW_SHARED: case OP_REBORROW_EXCLUSIVE:
            nvm2c_printf(&b," refs[%u]=refs[%u]; refs[%u].region=region; refs[%u].exclusive=%u; refs[%u].parent=%u;\n",
                local,in->operands[1].u16,local,local,op==OP_REBORROW_EXCLUSIVE,local,in->operands[1].u16);break;
        case OP_REF_GET:
            nvm2c_printf(&b," { nown_record *record=nown_referent(origins,generations,&refs[%u]); if(!record){status=3;goto cleanup;} t[%d]=record->fields[%u]; }\n",local,n,in->operands[1].u16);break;
        case OP_REF_SET:
            nvm2c_printf(&b," { nown_record *record=nown_referent(origins,generations,&refs[%u]); if(!record){status=3;goto cleanup;} record->fields[%u]=t[%d]; t[%d]=(nown_value){0}; }\n",local,in->operands[1].u16,n-1,n-1);break;
        case OP_NOP: break;
        case OP_PUSH_VOID:nvm2c_printf(&b," t[%d]=(nown_value){0};\n",n);break;
        case OP_PUSH_I64:
            nvm2c_printf(&b,shared?" t[%d]=(nown_value){.scalar=nown_i64_bits(UINT64_C(%llu)),.tag=%u};\n":
                " t[%d]=(nown_value){.scalar=(int64_t)UINT64_C(%llu),.tag=%u};\n",n,(unsigned long long)(uint64_t)in->operands[0].i64,TAG_INT);break;
        case OP_PUSH_U8: case OP_PUSH_BOOL:
            nvm2c_printf(&b," t[%d]=(nown_value){.scalar=%u,.tag=%u};\n",n,in->operands[0].u8,op==OP_PUSH_U8?TAG_U8:TAG_BOOL);break;
        case OP_PUSH_F64: {
            uint64_t bits;memcpy(&bits,&in->operands[0].f64,sizeof(bits));
            nvm2c_printf(&b," t[%d]=(nown_value){.floating=nown_float_bits(UINT64_C(0x%016llx)),.tag=%u};\n",n,(unsigned long long)bits,TAG_FLOAT);
            break;
        }
        case OP_PUSH_STR: {
            if(!value_graph) goto fail;
            uint32_t index=in->operands[0].u32;
            if(index>=mod->string_count || !mod->strings || !mod->string_lengths ||
               !mod->strings[index] ||
               memchr(mod->strings[index],'\0',mod->string_lengths[index])) goto fail;
            nvm2c_puts(&b," a.string=nown_string_new((const unsigned char *)");
            emit_c_string_lit(&b,mod->strings[index],mod->string_lengths[index]);
            nvm2c_printf(&b,",%u); if(!a.string){status=1;goto cleanup;} a.tag=%u; t[%d]=a; a=(nown_value){0};\n",mod->string_lengths[index],TAG_STRING,n);
            break;
        }
        case OP_LOAD_LOCAL:
            nvm2c_printf(&b," if(!nown_retain(l[%u])){status=1;goto cleanup;} t[%d]=l[%u];\n",local,n,local);break;
        case OP_STORE_LOCAL: case OP_OWN_STORE_LOCAL:
            nvm2c_printf(&b," nown_release(l[%u]); l[%u]=t[%d]; t[%d]=(nown_value){0};\n",local,local,n-1,n-1);break;
        case OP_OWN_MOVE_LOCAL:
            nvm2c_printf(&b," t[%d]=l[%u]; l[%u]=(nown_value){0};\n",n,local,local);break;
        case OP_OWN_PACK: {
            unsigned count=layouts.items[in->operands[0].u32].field_count;
            nvm2c_printf(&b," a.record=NOWN_ALLOC(1,sizeof(nown_record)+%u*sizeof(nown_value));\n if(!a.record){status=1;goto cleanup;} a.record->refs=1; a.record->count=%u; a.record->layout=%u;\n",count,count,in->operands[0].u32);
            for (unsigned f=0;f<count;f++)
                nvm2c_printf(&b," a.record->fields[%u]=t[%d]; t[%d]=(nown_value){0};\n",f,n-(int)count+(int)f,n-(int)count+(int)f);
            if(shared)nvm2c_puts(&b," a.category=1;\n");
            nvm2c_printf(&b," a.tag=%u; t[%d]=a; a=(nown_value){0};\n",TAG_STRUCT,n-(int)count);break;
        }
        case OP_OWN_UNPACK_LOCAL: {
            NvmAffineType type;if(!owned_emission_local(state,mixed,owner_arrays,function,local,&type)) goto fail;
            unsigned count=layouts.items[type.layout].field_count;
            nvm2c_printf(&b," a=l[%u]; l[%u]=(nown_value){0};\n",local,local);
            for (unsigned f=0;f<count;f++)
                nvm2c_printf(&b," t[%d]=a.record->fields[%u]; a.record->fields[%u]=(nown_value){0};\n",n+(int)f,f,f);
            nvm2c_puts(&b," nown_release(a); a=(nown_value){0};\n");break;
        }
        case OP_AGG_PACK: {
            if(!mixed || in->operands[0].u8!=AGG_RECORD)goto fail;
            NvmMixedRecordIdentity identity;
            if(!nvm_mixed_samples_record(mixed,in->operands[1].u32,&identity) || identity.field_count!=in->operands[3].u16)goto fail;
            unsigned count=identity.field_count;int base=n-(int)count;
            nvm2c_puts(&b," {NmsValue fields[256]={{0}};NmsHandle handle=0;\n");
            for(unsigned f=0;f<count;f++)
                nvm2c_printf(&b," if(!nown_to_managed(t[%d],&fields[%u])){status=3;goto cleanup;}\n",base+(int)f,f);
            nvm2c_printf(&b," status=nown_status(nms_record_create(managed,%u,fields,%u,&handle));if(status)goto cleanup;\n",identity.managed_record,count);
            for(unsigned f=0;f<count;f++)nvm2c_printf(&b," nown_release(t[%d]);t[%d]=(nown_value){0};\n",base+(int)f,base+(int)f);
            nvm2c_printf(&b," t[%d]=(nown_value){.tag=%u,.category=2,.managed=managed,.handle=handle};}\n",base,TAG_STRUCT);break;
        }
        case OP_ARR_NEW:case OP_ARR_LITERAL: {
            if(!shared)goto fail;
            unsigned count=op==OP_ARR_LITERAL?in->operands[1].u16:0;int base=n-(int)count;
            nvm2c_puts(&b," {NmsHandle handle=0;\n");
            if(op==OP_ARR_NEW)nvm2c_puts(&b," status=nown_status(nms_vm_array_create(managed,3,&handle));\n");
            else {
                nvm2c_puts(&b," uint64_t bits[256]={0};uint32_t tags[256]={0};\n");
                for(unsigned f=0;f<count;f++)nvm2c_printf(&b," if(t[%d].tag!=3 || t[%d].category){status=3;goto cleanup;}memcpy(&bits[%u],&t[%d].floating,8);tags[%u]=3;\n",base+(int)f,base+(int)f,f,base+(int)f,f);
                nvm2c_printf(&b," status=nown_status(nms_vm_array_literal(managed,3,bits,tags,%u,&handle));\n",count);
            }
            nvm2c_puts(&b," if(status)goto cleanup;\n");
            for(unsigned f=0;f<count;f++)nvm2c_printf(&b," t[%d]=(nown_value){0};\n",base+(int)f);
            nvm2c_printf(&b," t[%d]=(nown_value){.tag=%u,.category=2,.managed=managed,.handle=handle};}\n",base,TAG_ARRAY);break;
        }
        case OP_ARR_PUSH:case OP_ARR_SET:case OP_ARR_GET:case OP_ARR_LEN: {
            if(!shared)goto fail;
            int base=n-(op==OP_ARR_SET?3:op==OP_ARR_LEN?1:2);
            nvm2c_printf(&b," if(t[%d].category!=2 || t[%d].tag!=%u || t[%d].managed!=managed){status=3;goto cleanup;}\n",base,base,TAG_ARRAY,base);
            if(op==OP_ARR_GET || op==OP_ARR_SET)nvm2c_printf(&b," if(t[%d].tag!=%u || t[%d].category){status=3;goto cleanup;}\n",base+1,TAG_INT,base+1);
            if(op==OP_ARR_PUSH || op==OP_ARR_SET) {
                nvm2c_printf(&b," {NmsValue value={0};if(t[%d].tag!=3 || !nown_to_managed(t[%d],&value)){status=3;goto cleanup;}\n",n-1,n-1);
                if(op==OP_ARR_PUSH)nvm2c_printf(&b," status=nown_status(nms_value_array_append(managed,t[%d].handle,value));\n",base);
                else nvm2c_printf(&b," status=nown_status(nms_value_array_set(managed,t[%d].handle,(uint64_t)t[%d].scalar,value));\n",base,base+1);
                nvm2c_puts(&b," if(status)goto cleanup;}\n");
                for(int f=base+1;f<n;f++)nvm2c_printf(&b," t[%d]=(nown_value){0};\n",f);
            } else if(op==OP_ARR_GET) {
                nvm2c_printf(&b," {NmsValue value={0};status=nown_status(nms_value_array_get(managed,t[%d].handle,(uint64_t)t[%d].scalar,&value));if(status)goto cleanup;if(!nown_from_managed(managed,value,&a)){(void)nms_value_release(managed,value);status=3;goto cleanup;}nown_release(t[%d]);t[%d]=a;a=(nown_value){0};t[%d]=(nown_value){0};}\n",base,base+1,base,base,base+1);
            } else {
                nvm2c_printf(&b," {uint32_t length=0;status=nown_status(nms_value_array_length(managed,t[%d].handle,&length));if(status)goto cleanup;nown_release(t[%d]);t[%d]=(nown_value){.scalar=length,.tag=%u};}\n",base,base,base,TAG_INT);
            }
            break;
        }
        case OP_AGG_GET: case OP_STRUCT_GET:
            if(shared)nvm2c_printf(&b," if(t[%d].category==2){NmsValue value={0}; status=nown_status(nms_record_get(managed,t[%d].handle,%u,&value));if(status)goto cleanup; if(!nown_from_managed(managed,value,&a)){(void)nms_value_release(managed,value);status=3;goto cleanup;} nown_release(t[%d]);t[%d]=a;a=(nown_value){0};}else {\n",n-1,n-1,local,n-1,n-1);
            nvm2c_printf(&b," if(!nown_retain(t[%d].record->fields[%u])){status=1;goto cleanup;} a=t[%d]; t[%d]=a.record->fields[%u]; nown_release(a); a=(nown_value){0};\n",n-1,local,n-1,n-1,local);
            if(shared)nvm2c_puts(&b," }\n");
            break;
        case OP_DUP:nvm2c_printf(&b," if(!nown_retain(t[%d])){status=1;goto cleanup;} t[%d]=t[%d];\n",n-1,n,n-1);break;
        case OP_POP:nvm2c_printf(&b," nown_release(t[%d]); t[%d]=(nown_value){0};\n",n-1,n-1);break;
        case OP_ROT3:nvm2c_printf(&b," a=t[%d];t[%d]=t[%d];t[%d]=t[%d];t[%d]=a;a=(nown_value){0};\n",n-1,n-1,n-2,n-2,n-3,n-3);break;
        case OP_SWAP:nvm2c_printf(&b," a=t[%d]; t[%d]=t[%d]; t[%d]=a; a=(nown_value){0};\n",n-1,n-1,n-2,n-2);break;
        case OP_I64_NEG: case OP_NEG:
            if(shared)nvm2c_printf(&b," t[%d].scalar=nown_i64_bits(UINT64_C(0)-(uint64_t)t[%d].scalar);\n",n-1,n-1);
            else nvm2c_printf(&b," t[%d].scalar=(int64_t)(UINT64_C(0)-(uint64_t)t[%d].scalar);\n",n-1,n-1);
            break;
        case OP_BOOL_NOT: case OP_NOT:nvm2c_printf(&b," t[%d].scalar=!t[%d].scalar;\n",n-1,n-1);break;
        case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_ADD: case OP_SUB: case OP_MUL:
            nvm2c_printf(&b,shared?" t[%d].scalar=nown_i64_bits((uint64_t)t[%d].scalar %s (uint64_t)t[%d].scalar); t[%d]=(nown_value){0};\n":
                " t[%d].scalar=(int64_t)((uint64_t)t[%d].scalar %s (uint64_t)t[%d].scalar); t[%d]=(nown_value){0};\n",n-2,n-2,(op==OP_ADD || op==OP_I64_ADD)?"+":(op==OP_SUB || op==OP_I64_SUB)?"-":"*",n-1,n-1);break;
        case OP_I64_DIV_S: case OP_I64_REM_S: case OP_DIV: case OP_MOD:
            nvm2c_printf(&b," a=t[%d]; c=t[%d]; t[%d].scalar=!c.scalar?0:(a.scalar==INT64_MIN && c.scalar==-1)?%s:a.scalar %s c.scalar; t[%d]=(nown_value){0}; a=c=(nown_value){0};\n",n-2,n-1,n-2,(op==OP_DIV || op==OP_I64_DIV_S)?"INT64_MIN":"0",(op==OP_DIV || op==OP_I64_DIV_S)?"/":"%",n-1);break;
        case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV: {
            if(shared)nvm2c_printf(&b," if(t[%d].tag!=%u || t[%d].tag!=%u){status=3;goto cleanup;}\n",n-2,TAG_FLOAT,n-1,TAG_FLOAT);
            const char *operation=op==OP_F64_ADD?"add":op==OP_F64_SUB?"sub":op==OP_F64_MUL?"mul":"div";
            nvm2c_printf(&b," t[%d]=(nown_value){.floating=nano_rt_f64_%s(t[%d].floating,t[%d].floating),.tag=%u}; t[%d]=(nown_value){0};\n",n-2,operation,n-2,n-1,TAG_FLOAT,n-1);break;
        }
        case OP_F64_NEG:
            if(shared)nvm2c_printf(&b," if(t[%d].tag!=%u){status=3;goto cleanup;}\n",n-1,TAG_FLOAT);
            nvm2c_printf(&b," t[%d].floating=-t[%d].floating;\n",n-1,n-1);break;
        case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE: {
            if(shared)nvm2c_printf(&b," if(t[%d].tag!=%u || t[%d].tag!=%u){status=3;goto cleanup;}\n",n-2,TAG_FLOAT,n-1,TAG_FLOAT);
            const char *operation=op==OP_F64_EQ?"==":op==OP_F64_NE?"!=":op==OP_F64_LT?"<":op==OP_F64_LE?"<=":op==OP_F64_GT?">":">=";
            nvm2c_printf(&b," t[%d]=(nown_value){.scalar=(t[%d].floating %s t[%d].floating),.tag=%u}; t[%d]=(nown_value){0};\n",n-2,n-2,operation,n-1,TAG_BOOL,n-1);break;
        }
        case OP_I64_EQ:case OP_I64_NE:case OP_I64_LT_S:case OP_I64_LE_S:case OP_I64_GT_S:case OP_I64_GE_S:
        case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE: {
            if(op>=OP_I64_EQ && op<=OP_I64_GE_S)op=(uint8_t)(OP_EQ+(op-OP_I64_EQ));
            const char *operation=op==OP_EQ?"==":op==OP_NE?"!=":op==OP_LT?"<":op==OP_LE?"<=":op==OP_GT?">":">=";
            nvm2c_puts(&b," { int comparison; ");
            if(shared && (op==OP_EQ || op==OP_NE))nvm2c_printf(&b," if(t[%d].tag!=t[%d].tag)comparison=%u;else if(t[%d].tag==0)comparison=%u;else ",n-2,n-1,op==OP_NE,n-2,op==OP_EQ);
            if(op==OP_EQ || op==OP_NE)
                nvm2c_printf(&b," if(t[%d].tag==%u) comparison=%snown_string_equal(t[%d].string,t[%d].string); else ",n-2,TAG_STRING,op==OP_NE?"!":"",n-2,n-1);
            nvm2c_printf(&b," if(t[%d].tag==%u) {\n",n-2,TAG_FLOAT);
            if(op==OP_EQ || op==OP_NE)
                nvm2c_printf(&b," comparison=t[%d].floating %s t[%d].floating;\n",n-2,operation,n-1);
            else
                nvm2c_printf(&b," int order=t[%d].floating<t[%d].floating?-1:t[%d].floating>t[%d].floating?1:0; comparison=order %s 0;\n",n-2,n-1,n-2,n-1,operation);
            nvm2c_printf(&b," } else comparison=t[%d].scalar %s t[%d].scalar; nown_release(t[%d]); nown_release(t[%d]); t[%d]=(nown_value){.scalar=comparison,.tag=%u}; t[%d]=(nown_value){0}; }\n",n-2,operation,n-1,n-2,n-1,n-2,TAG_BOOL,n-1);break;
        }
        case OP_BOOL_AND:case OP_BOOL_OR:case OP_AND: case OP_OR:
            nvm2c_printf(&b," t[%d]=(nown_value){.scalar=(t[%d].scalar %s t[%d].scalar),.tag=%u}; t[%d]=(nown_value){0};\n",n-2,n-2,(op==OP_AND || op==OP_BOOL_AND)?"&&":"||",n-1,TAG_BOOL,n-1);break;
        case OP_JMP:nvm2c_printf(&b," goto L%u;\n",d->resolved_target-fn->code_offset);continue;
        case OP_JMP_TRUE: case OP_JMP_FALSE:
            nvm2c_printf(&b," a=t[%d]; t[%d]=(nown_value){0}; if(%sa.scalar) {a=(nown_value){0};goto L%u;} a=(nown_value){0};\n",n-1,n-1,op==OP_JMP_TRUE?"":"!",d->resolved_target-fn->code_offset);break;
        case OP_ASSERT:
            nvm2c_printf(&b," a=t[%d]; t[%d]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};\n",n-1,n-1);break;
        case OP_PRINT: case OP_PRINTLN:
            if(!value_graph) goto fail;
            if(owner_arrays)nvm2c_printf(&b," if(t[%d].tag!=%u || t[%d].category){status=3;goto cleanup;} (void)fprintf(stdout,\"%%lld\",(long long)t[%d].scalar);",n-1,TAG_INT,n-1,n-1);
            else nvm2c_printf(&b," (void)fwrite(t[%d].string->bytes,1,t[%d].string->length,stdout);",n-1,n-1);
            if(op==OP_PRINTLN) nvm2c_puts(&b," (void)fputc('\\n',stdout);");
            nvm2c_printf(&b," nown_release(t[%d]); t[%d]=(nown_value){0};\n",n-1,n-1);break;
        case OP_RET:
            if(shared && result_type.tag!=TAG_VOID)nvm2c_printf(&b," if(t[0].tag!=%u || t[0].category!=%u){status=3;goto cleanup;}\n",result_type.tag,result_type.tag==TAG_STRUCT?1u:0u);
            if(result_type.tag==TAG_STRUCT)
                nvm2c_printf(&b," if(!t[0].record || t[0].record->layout!=%u || t[0].record->count!=%u){status=3;goto cleanup;}\n",result_type.layout,result_fields);
            if(result_type.tag!=TAG_VOID) nvm2c_puts(&b," pending=t[0]; t[0]=(nown_value){0};\n");
            nvm2c_puts(&b," goto cleanup;\n");continue;
        default:goto fail;
        }
        nvm2c_printf(&b," goto L%u;\n",d->next_byte_offset);
    }
    nvm2c_puts(&b,"cleanup:;\n for(size_t i=0;i<256;i++){nown_release(t[i]);nown_release(l[i]);}\n");
    if(shared)nvm2c_puts(&b," nown_release(a);nown_release(c);a=c=(nown_value){0};\n");
    if(shared && !function)nvm2c_puts(&b," if(managed->live_objects || managed->live_bytes)status=3;\n if(managed->active)(void)nms_finish(managed,status?NMS_STATE:NMS_OK,0);\n if(nms_dispose(managed)!=NMS_OK)status=3;\n");
    if(result_type.tag!=TAG_VOID) nvm2c_puts(&b,consuming?
        " if(!status){*result=pending; pending=(nown_value){0};}\n":
        " if(!status){*result=pending.scalar; pending=(nown_value){0};}\n");
    nvm2c_puts(&b," nown_release(pending); return status;\n}\n");
    if(!function) nvm2c_puts(&b,"#ifndef NVM2C_NO_MAIN\nint main(void){int64_t result=0;return nvm_owned_entry(&result)?1:(int)result;}\n#endif\n");
    free(depth);free(queue);nvm_affine_state_free(state);nvm_v2_layouts_free(&layouts);vm_decoded_function_free(&code);
    if (b.failed) {free(b.data);return NULL;}return b.data;
fail:
    free(depth);free(queue);nvm_affine_state_free(state);nvm_v2_layouts_free(&layouts);vm_decoded_function_free(&code);free(b.data);
    if(err && err_len)snprintf(err,err_len,"I cannot lower this verified owned-transfer function");
    return NULL;
}

static char *emit_owned_module(const NvmModule *mod,char *err,size_t err_len) {
    return emit_owned_function(mod,0,NULL,NULL,err,err_len);
}

/* I select only a fresh complete mixed runtime conjunction. */
static char *emit_mixed_samples_module(const NvmModule *mod,char *err,size_t err_len) {
    NvmMixedSamplesPlan *plan=NULL;
    NvmMixedShapeResult result=nvm_mixed_samples_admit(mod,&plan);
    if(result.status!=NVM_MIXED_SHAPE_PROVED){if(err && err_len)snprintf(err,err_len,"%s",result.message);return NULL;}
    NvmMixedSignature entry;
    if(!nvm_mixed_samples_signature(plan,0,&entry) ||
       (entry.result.tag!=TAG_INT && entry.result.tag!=TAG_BOOL && entry.result.tag!=TAG_U8)) {
        nvm_mixed_samples_plan_free(plan);
        if(err && err_len)snprintf(err,err_len,"I require a scalar mixed entry result");
        return NULL;
    }
    char *source=emit_owned_function(mod,0,plan,NULL,err,err_len);
    nvm_mixed_samples_plan_free(plan);return source;
}

#ifdef NANO_OWNED_ARRAY_PRIVATE_RUNTIME
bool nvm2c_emit_owned_array_private(const NvmModule *mod,char **out,char *err,size_t err_len) {
    if(!out)return false;
    NvmOwnedArrayPlan *plan=NULL;
    NvmOwnerAuthorityResult checked=nvm_prepare_owned_array_authority(mod,&plan);
    if(checked.status!=NVM_OWNER_AUTH_PREPARED) {
        if(err && err_len)snprintf(err,err_len,"%s",checked.message);
        return false;
    }
    NvmOwnerSignature entry;
    if(!nvm_owned_array_plan_signature(plan,0,&entry) || entry.parameter_count ||
       (entry.result.tag!=TAG_INT && entry.result.tag!=TAG_BOOL && entry.result.tag!=TAG_U8)) {
        nvm_owned_array_plan_free(plan);
        if(err && err_len)snprintf(err,err_len,"I require a scalar private owner ARRAY entry.");
        return false;
    }
    char *source=emit_owned_function(mod,0,NULL,plan,err,err_len);
    nvm_owned_array_plan_free(plan);
    if(!source)return false;
    *out=source;return true;
}
#endif
