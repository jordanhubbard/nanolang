/* I call unchanged effect steps and identity helpers without product C edits. */
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main

void nvm2c_affine_owner_state_snapshot(const NvmAffineState *s);
static void effect_identity(Reader *r) {
    uint32_t present=word(r),a=word(r),b=word(r),size=word(r);size_t at=0;NvmModule model={0};
    if(!r->ok || present!=1 || !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(r->pos==r->size)printf("%u",effect_key_equal(present?&model:NULL,a,b));else r->ok=false;
    free_model(&model);
}
static void effect_transfer(Reader *r) {
    uint32_t opcode=word(r),a=word(r),b=word(r),function=word(r),present=word(r),count=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    if(!r->ok || (opcode && opcode!=OP_HANDLER_PUSH && opcode!=OP_HANDLER_POP && opcode!=OP_PERFORM) ||
       present>1 || (!opcode && !present) || b>UINT16_MAX || (opcode==OP_HANDLER_POP && a>UINT16_MAX) || count>NVM_OWNERSHIP_MAX_EFFECTS ||
       !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);
    uint32_t slot=function<NVM_OWNED_MAX_FUNCTIONS?function:0;
    uint32_t before[NVM_OWNERSHIP_MAX_EFFECTS];
    calls->escape_count[slot]=(uint16_t)count;
    for(uint32_t i=0;i<count;i++)calls->escapes[slot][i]=before[i]=word(r);
    if(!r->ok || !model.function_count){r->ok=false;free(calls);free_model(&model);return;}
    NvmAffineState *initial=nvm_affine_state_create(&model,0,model.functions[0].local_count);
    if(!initial){r->ok=false;free(calls);free_model(&model);return;}
    Frame *input=read_frame(r,initial),*out=NULL;nvm_affine_state_free(initial);
    if(r->ok && r->pos==r->size) {
        out=frame_clone(input);if(!out)exit(2);
        const char *error=NULL;bool valid;
        if(!opcode)valid=effect_escape(calls,present?&model:NULL,function,out,a);
        else {
            DecodedInstruction instruction={0};instruction.opcode=(uint8_t)opcode;
            if(opcode==OP_HANDLER_POP)instruction.operands[0].u16=(uint16_t)a;else instruction.operands[0].u32=a;
            instruction.operands[1].u16=(uint16_t)b;
            error=step(out,&instruction,model.functions[0].local_count,present?&model:NULL,function,0,calls);valid=error==NULL;
        }
        if(!valid){frame_free(out);out=frame_clone(input);if(!out)exit(2);calls->escape_count[slot]=(uint16_t)count;memcpy(calls->escapes[slot],before,count*sizeof(*before));}
        printf("%u|%s|",valid,error?error:"");snapshot(out,&model);putchar('|');nvm2c_affine_owner_state_snapshot(out->locals);putchar('|');
        for(uint16_t i=0;i<calls->escape_count[slot];i++)number(calls->escapes[slot][i]);
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
        else if(kind==11)effect_transfer(&r);else if(kind==12)effect_identity(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
