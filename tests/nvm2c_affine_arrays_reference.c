/* I call the unchanged production array step with checked state and module data. */
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main

static void array_transfer(Reader *r) {
    uint32_t opcode=word(r),tag=word(r),count=word(r),function=word(r),offset=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    if(!r->ok || opcode<OP_ARR_NEW || opcode>OP_ARR_LITERAL || tag>255 || count>UINT16_MAX ||
       !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(!model.function_count){free_model(&model);r->ok=false;return;}
    NvmAffineState *initial=nvm_affine_state_create(&model,0,model.functions[0].local_count);
    if(!initial){free_model(&model);r->ok=false;return;}
    Frame *input=read_frame(r,initial),*out=NULL;nvm_affine_state_free(initial);
    if(r->ok && r->pos==r->size) {
        out=frame_clone(input);if(!out)exit(2);
        AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);
        DecodedInstruction instruction={0};instruction.opcode=(uint8_t)opcode;
        instruction.operands[0].u8=(uint8_t)tag;instruction.operands[1].u16=(uint16_t)count;
        const char *error=step(out,&instruction,model.functions[0].local_count,&model,function,offset,calls);free(calls);
        if(error){frame_free(out);out=frame_clone(input);if(!out)exit(2);}
        printf("%u|%s|",error==NULL,error?error:"");snapshot(out,&model);
    } else r->ok=false;
    frame_free(input);frame_free(out);free_model(&model);
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
        else if(kind==8)array_transfer(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
