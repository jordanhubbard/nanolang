/* I call the unchanged production step with explicit checked global facts. */
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main

static void global_transfer(Reader *r) {
    uint32_t opcode=word(r),index=word(r),graph=word(r),effect=word(r),count=word(r);
    AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);
    uint8_t allowed[NVM_OWNERSHIP_MAX_GLOBALS]={0};
    if(!r->ok || (opcode!=OP_LOAD_GLOBAL && opcode!=OP_STORE_GLOBAL) ||
       graph>1 || effect>1 || count>NVM_OWNERSHIP_MAX_GLOBALS){r->ok=false;free(calls);return;}
    calls->value_graph=graph!=0;calls->effect_globals=effect!=0;calls->global_count=count;
    for(uint32_t i=0;i<count;i++) {
        uint32_t tag=word(r),mutable=word(r),layout=word(r),mask=word(r);
        if(!r->ok || tag>255 || mutable>1 || mask>3){r->ok=false;free(calls);return;}
        calls->globals[i]=(NvmOwnershipGlobal){.tag=(uint8_t)tag,.mutable=mutable!=0,.layout=layout};
        calls->global_allowed[0][i]=allowed[i]=(uint8_t)mask;
    }
    uint32_t size=word(r);size_t at=0;NvmModule model={0};
    if(!r->ok || !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)) {
        r->ok=false;free(calls);return;
    }
    if(!model.function_count){r->ok=false;free_model(&model);free(calls);return;}
    NvmAffineState *initial=nvm_affine_state_create(&model,0,model.functions[0].local_count);
    if(!initial){r->ok=false;free_model(&model);free(calls);return;}
    Frame *input=read_frame(r,initial),*out=NULL;nvm_affine_state_free(initial);
    if(r->ok && r->pos==r->size && input->global_count==count) {
        out=frame_clone(input);if(!out)exit(2);
        DecodedInstruction instruction={0};instruction.opcode=(uint8_t)opcode;instruction.operands[0].u32=index;
        const char *error=step(out,&instruction,model.functions[0].local_count,&model,0,0,calls);
        /* I publish no successor or callee precondition from a failed step. */
        if(error){frame_free(out);out=frame_clone(input);if(!out)exit(2);memcpy(calls->global_allowed[0],allowed,count);}
        printf("%u|%s|",error==NULL,error?error:"");snapshot(out,&model);putchar('|');
        for(uint32_t i=0;i<count;i++)printf("%u,",calls->global_allowed[0][i]);
    } else r->ok=false;
    frame_free(input);frame_free(out);free_model(&model);free(calls);
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
        else if(kind==7)global_transfer(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
