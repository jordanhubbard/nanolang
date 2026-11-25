/* I call the unchanged production aggregate step with checked state and module data. */
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main

void nvm2c_affine_owner_state_snapshot(const NvmAffineState *s);
static void aggregate_transfer(Reader *r) {
    uint32_t opcode=word(r),graph=word(r),a=word(r),b=word(r),c=word(r),d=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    bool handled=opcode==OP_STRUCT_GET || opcode==OP_MATCH_TAG || opcode==OP_AGG_PACK || opcode==OP_AGG_GET || opcode==OP_AGG_TAG;
    if(!r->ok || !handled || graph>1 || (opcode==OP_AGG_PACK ? a>255 : a>UINT16_MAX) || c>UINT16_MAX || d>UINT16_MAX ||
       !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(!model.function_count){free_model(&model);r->ok=false;return;}
    NvmAffineState *initial=nvm_affine_state_create(&model,0,model.functions[0].local_count);
    if(!initial){free_model(&model);r->ok=false;return;}
    Frame *input=read_frame(r,initial),*out=NULL;nvm_affine_state_free(initial);
    if(r->ok && r->pos==r->size) {
        out=frame_clone(input);if(!out)exit(2);
        AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);
        DecodedInstruction instruction={0};instruction.opcode=(uint8_t)opcode;
        if(opcode==OP_AGG_PACK) {
            instruction.operands[0].u8=(uint8_t)a;instruction.operands[1].u32=b;
            instruction.operands[2].u16=(uint16_t)c;instruction.operands[3].u16=(uint16_t)d;
        } else instruction.operands[0].u16=(uint16_t)a;
        calls->value_graph=graph!=0;
        const char *error=step(out,&instruction,model.functions[0].local_count,&model,0,0,calls);free(calls);
        if(error){frame_free(out);out=frame_clone(input);if(!out)exit(2);}
        printf("%u|%s|",error==NULL,error?error:"");snapshot(out,&model);putchar('|');nvm2c_affine_owner_state_snapshot(out->locals);
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
        else if(kind==10)aggregate_transfer(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
