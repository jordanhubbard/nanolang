/* I inspect unchanged CALL application with cached facts and uncached controls. */
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main

void nvm2c_affine_owner_state_snapshot(const NvmAffineState *s);
static void value_call(Reader *r) {
    uint32_t target=word(r),caller=word(r),graph=word(r),effect=word(r),cached=word(r),ok=word(r),globals=word(r),caller_count=word(r),callee_count=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    if(!r->ok || graph>1 || effect>1 || cached>1 || ok>1 || globals>NVM_OWNERSHIP_MAX_GLOBALS ||
       caller_count>NVM_OWNERSHIP_MAX_EFFECTS || callee_count>NVM_OWNERSHIP_MAX_EFFECTS ||
       !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(!model.function_count || model.function_count>NVM_OWNED_MAX_FUNCTIONS || caller>=model.function_count || (target>0 && target<model.function_count && caller==target)){r->ok=false;free_model(&model);return;}
    AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);
    uint8_t before_allowed[NVM_OWNERSHIP_MAX_GLOBALS],callee_allowed[NVM_OWNERSHIP_MAX_GLOBALS],callee_returns[NVM_OWNERSHIP_MAX_GLOBALS];
    uint32_t before_keys[NVM_OWNERSHIP_MAX_EFFECTS],callee_keys[NVM_OWNERSHIP_MAX_EFFECTS],declared=0;
    if(nvm_ownership_globals(&model,calls->globals,NVM_OWNERSHIP_MAX_GLOBALS,&declared)!=NVM_V2_OK || declared!=globals)r->ok=false;
    for(uint32_t i=0;r->ok && i<globals;i++) {
        uint32_t a=word(r),b=word(r),c=word(r);if(a>3 || b>3 || c>15){r->ok=false;break;}
        calls->global_allowed[caller][i]=before_allowed[i]=(uint8_t)a;callee_allowed[i]=(uint8_t)b;callee_returns[i]=(uint8_t)c;
    }
    for(uint32_t i=0;r->ok && i<caller_count;i++)calls->escapes[caller][i]=before_keys[i]=word(r);
    for(uint32_t i=0;r->ok && i<callee_count;i++)callee_keys[i]=word(r);
    calls->escape_count[caller]=(uint16_t)caller_count;calls->globals_valid=true;calls->value_graph=graph!=0;calls->effect_globals=effect!=0;calls->global_count=globals;
    bool bounded=target>0 && target<model.function_count;
    if(cached && bounded) {
        calls->status[target]=2;calls->results[target].ok=ok!=0;calls->escape_count[target]=(uint16_t)callee_count;
        memcpy(calls->escapes[target],callee_keys,callee_count*sizeof(*callee_keys));
        memcpy(calls->global_allowed[target],callee_allowed,globals);memcpy(calls->global_returns[target],callee_returns,globals);
    }
    NvmAffineState *initial=r->ok?nvm_affine_state_create(&model,caller,model.functions[caller].local_count):NULL;
    if(!initial){r->ok=false;free(calls);free_model(&model);return;}
    Frame *input=read_frame(r,initial),*out=NULL;nvm_affine_state_free(initial);
    if(r->ok && r->pos==r->size && input->global_count==globals) {
        out=frame_clone(input);if(!out)exit(2);
        DecodedInstruction instruction={0};instruction.opcode=OP_CALL;instruction.operands[0].u32=target;
        const char *error=step(out,&instruction,model.functions[caller].local_count,&model,caller,0,calls);
        bool derived=!cached && bounded && calls->status[target]==2;
        if(derived) {
            assert(calls->results[target].ok==(ok!=0));assert(calls->escape_count[target]==callee_count);
            assert(!callee_count || !memcmp(calls->escapes[target],callee_keys,callee_count*sizeof(*callee_keys)));
            assert(!globals || (!memcmp(calls->global_allowed[target],callee_allowed,globals) && !memcmp(calls->global_returns[target],callee_returns,globals)));
        }
        if(error){frame_free(out);out=frame_clone(input);if(!out)exit(2);memcpy(calls->global_allowed[caller],before_allowed,globals);calls->escape_count[caller]=(uint16_t)caller_count;memcpy(calls->escapes[caller],before_keys,caller_count*sizeof(*before_keys));}
        printf("%u|%s|",error==NULL,error?error:"");NvmModule view=model;view.functions=&model.functions[caller];snapshot(out,&view);putchar('|');nvm2c_affine_owner_state_snapshot(out->locals);putchar('|');
        for(uint32_t i=0;i<globals;i++){number(calls->global_allowed[caller][i]);}putchar('|');
        for(uint16_t i=0;i<calls->escape_count[caller];i++){number(calls->escapes[caller][i]);}printf("|%u",derived);
    } else r->ok=false;
    frame_free(input);frame_free(out);free(calls);free_model(&model);
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        FILE *file=fopen(argv[i],"rb");if(!file){puts("D");continue;}
        if(fseek(file,0,SEEK_END)){fclose(file);puts("D");continue;}
        long length=ftell(file);if(length<4 || fseek(file,0,SEEK_SET)){fclose(file);puts("D");continue;}
        uint8_t *bytes=malloc((size_t)length);if(!bytes)exit(2);
        bool loaded=fread(bytes,1,(size_t)length,file)==(size_t)length;fclose(file);
        Reader r={bytes,(size_t)length,0,loaded};uint32_t kind=word(&r);
        if(kind==0)flows(&r);else if(kind==1)queue(&r);else if(kind==2)frames(&r);else if(kind==3)profile(&r);
        else if(kind==13)value_call(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
