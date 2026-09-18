/* I lower the already verified standalone ownership contract to direct C
 * operations and constant-index temporaries. I emit no bytecode interpreter. */
static char *emit_owned_function(const NvmModule *mod,uint32_t function,char *err,size_t err_len) {
    Nvm2cBuf b={0}; b.err=err; b.err_len=err_len;
    VmDecodedFunction code={0}; char decode_error[VM_DECODE_ERROR_SIZE];
    NvmAffineState *state=NULL; NvmV2Layouts layouts={0};
    int *depth=NULL; uint32_t *queue=NULL;
    const NvmFunctionEntry *fn=&mod->functions[function];
    if (!vm_decode_function(mod,function,&code,decode_error) ||
        nvm_v2_layouts_decode(mod->layout_data,mod->layout_size,&layouts)!=NVM_V2_OK ||
        !(state=nvm_affine_state_create(mod,function,fn->local_count))) goto fail;
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
            if (!nvm_affine_local_type(state,in->operands[0].u16,&type)) goto fail;
            pop=0;push=layouts.items[type.layout].field_count;
        }
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
    if (!function) nvm2c_puts(&b,
        "/* I transfer unique owners; field observations retain only their shell. */\n"
        "#include <stdint.h>\n#include <stdlib.h>\n#include <stddef.h>\n#include <limits.h>\n"
        "#ifndef NOWN_ALLOC\n#define NOWN_ALLOC calloc\n#endif\n"
        "#ifndef NOWN_FREE\n#define NOWN_FREE free\n#endif\n"
        "typedef struct nown_record nown_record;\n"
        "typedef struct { int64_t scalar; nown_record *record; } nown_value;\n"
        "struct nown_record { size_t refs, count; nown_value fields[]; };\n"
        "typedef struct { unsigned root,region,exclusive,parent,depth,origin; uint64_t generation; uint16_t fields[32]; } nown_reference;\n"
        "static nown_record *nown_referent(nown_value *const origins[2],const uint64_t generations[2],const nown_reference *ref) {\n"
        " if(ref->origin>1 || !origins[ref->origin] || ref->generation!=generations[ref->origin])return NULL;\n"
        " nown_record *record=origins[ref->origin][ref->root].record;\n"
        " for(unsigned i=0;i<ref->depth;i++) record=record->fields[ref->fields[i]].record;\n"
        " return record;\n}\n"
        "static void nown_release(nown_value v) {\n"
        " if (v.record && --v.record->refs==0) {\n"
        "  for(size_t i=0;i<v.record->count;i++) nown_release(v.record->fields[i]);\n"
        "  NOWN_FREE(v.record);\n }\n}\n"
        "static void nown_retain(nown_value v) { if(v.record) ++v.record->refs; }\n"
        "/* I return status separately so allocation failure still cleans every root. */\n");
    if (!function && mod->function_count==2) {
        char *helper=emit_owned_function(mod,1,err,err_len);
        if(!helper) goto fail;
        nvm2c_puts(&b,helper);free(helper);
    }
    nvm2c_puts(&b,function?
        "static int nown_helper(nown_value *origin,const nown_reference *borrowed,uint64_t caller_generation,uint64_t generation,int64_t *result) {\n":
        "int nvm_owned_entry(int64_t *result) {\n");
    nvm2c_puts(&b,
        " nown_value t[256]={{0}}, l[256]={{0}}, a={0}, c={0};\n"
        " nown_reference refs[256]={{0}}; unsigned region=0;\n"
        " int status=0; (void)a; (void)c; (void)nown_retain; (void)refs; (void)region; (void)nown_referent;\n");
    if(function) {
        nvm2c_puts(&b," nown_value *origins[2]={origin,l}; uint64_t generations[2]={caller_generation,generation};\n");
        for(uint16_t p=0;p<fn->arity;p++) {
            NvmAffineType param;NvmReferenceMode mode;
            if(!nvm_affine_parameter_at(state,p,&param,&mode)) goto fail;
            nvm2c_printf(&b," refs[%u]=borrowed[%u]; refs[%u].region=0; refs[%u].parent=UINT16_MAX; refs[%u].exclusive=%u;\n",p,p,p,p,p,mode==NVM_REFERENCE_EXCLUSIVE);
        }
    } else nvm2c_puts(&b," nown_value *origins[2]={l,NULL}; uint64_t generations[2]={1,0},next_generation=1; (void)next_generation;\n");
    nvm2c_puts(&b," (void)origins; (void)generations; goto L0;\n");
    for (uint32_t i=0;i<code.instruction_count;i++) {
        if (depth[i]<0) continue;
        const VmDecodedInstruction *d=&code.instructions[i];
        const DecodedInstruction *in=&d->instruction;uint8_t op=in->opcode;int n=depth[i];
        unsigned local=in->operands[0].u16;
        nvm2c_printf(&b,"L%u:;\n",d->byte_offset);
        switch(op) {
        case OP_CALL_REF:
            nvm2c_printf(&b," if(next_generation==UINT64_MAX){status=3;goto cleanup;}\n status=nown_helper(l,&refs[%u],generations[0],++next_generation,&t[%d].scalar); if(status)goto cleanup;\n",in->operands[1].u16,n);break;
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
            nvm2c_printf(&b," refs[%u].origin=%u; refs[%u].generation=generations[%u];\n",local,function,local,function);
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
        case OP_PUSH_I64:
            nvm2c_printf(&b," t[%d]=(nown_value){(int64_t)UINT64_C(%llu),NULL};\n",n,(unsigned long long)(uint64_t)in->operands[0].i64);break;
        case OP_PUSH_U8: case OP_PUSH_BOOL:
            nvm2c_printf(&b," t[%d]=(nown_value){%u,NULL};\n",n,in->operands[0].u8);break;
        case OP_LOAD_LOCAL:
            nvm2c_printf(&b," t[%d]=l[%u]; nown_retain(t[%d]);\n",n,local,n);break;
        case OP_STORE_LOCAL: case OP_OWN_STORE_LOCAL:
            nvm2c_printf(&b," nown_release(l[%u]); l[%u]=t[%d]; t[%d]=(nown_value){0};\n",local,local,n-1,n-1);break;
        case OP_OWN_MOVE_LOCAL:
            nvm2c_printf(&b," t[%d]=l[%u]; l[%u]=(nown_value){0};\n",n,local,local);break;
        case OP_OWN_PACK: {
            unsigned count=layouts.items[in->operands[0].u32].field_count;
            nvm2c_printf(&b," a.record=NOWN_ALLOC(1,sizeof(nown_record)+%u*sizeof(nown_value));\n if(!a.record){status=1;goto cleanup;} a.record->refs=1; a.record->count=%u;\n",count,count);
            for (unsigned f=0;f<count;f++)
                nvm2c_printf(&b," a.record->fields[%u]=t[%d]; t[%d]=(nown_value){0};\n",f,n-(int)count+(int)f,n-(int)count+(int)f);
            nvm2c_printf(&b," t[%d]=a; a=(nown_value){0};\n",n-(int)count);break;
        }
        case OP_OWN_UNPACK_LOCAL: {
            NvmAffineType type;if(!nvm_affine_local_type(state,local,&type)) goto fail;
            unsigned count=layouts.items[type.layout].field_count;
            nvm2c_printf(&b," a=l[%u]; l[%u]=(nown_value){0};\n",local,local);
            for (unsigned f=0;f<count;f++)
                nvm2c_printf(&b," t[%d]=a.record->fields[%u]; a.record->fields[%u]=(nown_value){0};\n",n+(int)f,f,f);
            nvm2c_puts(&b," nown_release(a); a=(nown_value){0};\n");break;
        }
        case OP_AGG_GET: case OP_STRUCT_GET:
            nvm2c_printf(&b," a=t[%d]; t[%d]=a.record->fields[%u]; nown_retain(t[%d]); nown_release(a); a=(nown_value){0};\n",n-1,n-1,local,n-1);break;
        case OP_DUP:nvm2c_printf(&b," t[%d]=t[%d];\n",n,n-1);break;
        case OP_POP:nvm2c_printf(&b," t[%d]=(nown_value){0};\n",n-1);break;
        case OP_SWAP:nvm2c_printf(&b," a=t[%d]; t[%d]=t[%d]; t[%d]=a; a=(nown_value){0};\n",n-1,n-1,n-2,n-2);break;
        case OP_NEG:nvm2c_printf(&b," t[%d].scalar=(int64_t)(UINT64_C(0)-(uint64_t)t[%d].scalar);\n",n-1,n-1);break;
        case OP_NOT:nvm2c_printf(&b," t[%d].scalar=!t[%d].scalar;\n",n-1,n-1);break;
        case OP_ADD: case OP_SUB: case OP_MUL:
            nvm2c_printf(&b," t[%d].scalar=(int64_t)((uint64_t)t[%d].scalar %s (uint64_t)t[%d].scalar); t[%d]=(nown_value){0};\n",n-2,n-2,op==OP_ADD?"+":op==OP_SUB?"-":"*",n-1,n-1);break;
        case OP_DIV: case OP_MOD:
            nvm2c_printf(&b," a=t[%d]; c=t[%d]; t[%d].scalar=!c.scalar?0:(a.scalar==INT64_MIN && c.scalar==-1)?%s:a.scalar %s c.scalar; t[%d]=(nown_value){0}; a=c=(nown_value){0};\n",n-2,n-1,n-2,op==OP_DIV?"INT64_MIN":"0",op==OP_DIV?"/":"%",n-1);break;
        case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE: case OP_AND: case OP_OR: {
            const char *operation=op==OP_EQ?"==":op==OP_NE?"!=":op==OP_LT?"<":op==OP_LE?"<=":op==OP_GT?">":op==OP_GE?">=":op==OP_AND?"&&":"||";
            nvm2c_printf(&b," t[%d].scalar=(t[%d].scalar %s t[%d].scalar); t[%d]=(nown_value){0};\n",n-2,n-2,operation,n-1,n-1);break;
        }
        case OP_JMP:nvm2c_printf(&b," goto L%u;\n",d->resolved_target-fn->code_offset);continue;
        case OP_JMP_TRUE: case OP_JMP_FALSE:
            nvm2c_printf(&b," a=t[%d]; t[%d]=(nown_value){0}; if(%sa.scalar) {a=(nown_value){0};goto L%u;} a=(nown_value){0};\n",n-1,n-1,op==OP_JMP_TRUE?"":"!",d->resolved_target-fn->code_offset);break;
        case OP_ASSERT:
            nvm2c_printf(&b," a=t[%d]; t[%d]=(nown_value){0}; if(!a.scalar){status=2;goto cleanup;} a=(nown_value){0};\n",n-1,n-1);break;
        case OP_RET:nvm2c_puts(&b," *result=t[0].scalar; goto cleanup;\n");continue;
        default:goto fail;
        }
        nvm2c_printf(&b," goto L%u;\n",d->next_byte_offset);
    }
    nvm2c_puts(&b,"cleanup:;\n for(size_t i=0;i<256;i++){nown_release(t[i]);nown_release(l[i]);}\n return status;\n}\n");
    if(!function) nvm2c_puts(&b,"#ifndef NVM2C_NO_MAIN\nint main(void){int64_t result=0;return nvm_owned_entry(&result)?1:(int)result;}\n#endif\n");
    free(depth);free(queue);nvm_affine_state_free(state);nvm_v2_layouts_free(&layouts);vm_decoded_function_free(&code);
    if (b.failed) {free(b.data);return NULL;}return b.data;
fail:
    free(depth);free(queue);nvm_affine_state_free(state);nvm_v2_layouts_free(&layouts);vm_decoded_function_free(&code);free(b.data);
    if(err && err_len)snprintf(err,err_len,"I cannot lower this verified owned-transfer function");
    return NULL;
}

static char *emit_owned_module(const NvmModule *mod,char *err,size_t err_len) {
    return emit_owned_function(mod,0,err,err_len);
}
