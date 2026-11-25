/* I reuse the original frame reader and call the unchanged production step. */
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main

static bool scalar_handles(uint32_t opcode) {
    switch(opcode) {
    case OP_NOP: case OP_JMP: case OP_HALT:
    case OP_PUSH_VOID: case OP_PUSH_I64: case OP_PUSH_U8: case OP_PUSH_F64: case OP_PUSH_BOOL:
    case OP_DUP: case OP_POP: case OP_SWAP: case OP_PRINT: case OP_PRINTLN:
    case OP_ASSERT: case OP_JMP_TRUE: case OP_JMP_FALSE:
    case OP_NOT: case OP_AND: case OP_OR:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: case OP_NEG:
    case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV: case OP_F64_NEG:
    case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:return true;
    default:return false;
    }
}
static void scalar_profile(Reader *r) {
    uint32_t opcode=word(r);
    if(!r->ok || opcode>255 || r->pos!=r->size){r->ok=false;return;}
    putchar(scalar_handles(opcode)?'1':'0');
}
static void scalar_transfer(Reader *r) {
    uint32_t opcode=word(r),operand=word(r),graph=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    if(!r->ok || opcode>255 || operand>255 || graph>1 || !scalar_handles(opcode) ||
       !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(!model.function_count){free_model(&model);r->ok=false;return;}
    NvmAffineState *initial=nvm_affine_state_create(&model,0,model.functions[0].local_count);
    if(!initial){free_model(&model);r->ok=false;return;}
    Frame *input=read_frame(r,initial),*out=NULL;nvm_affine_state_free(initial);
    if(r->ok && r->pos==r->size) {
        out=frame_clone(input);if(!out)exit(2);
        AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);calls->value_graph=graph!=0;
        DecodedInstruction instruction={0};instruction.opcode=(uint8_t)opcode;instruction.operands[0].u8=(uint8_t)operand;
        const char *error=step(out,&instruction,model.functions[0].local_count,&model,0,0,calls);free(calls);
        /* I compare an atomic transfer result; a failed analysis has no successor. */
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
        else if(kind==4)scalar_profile(&r);else if(kind==5)scalar_transfer(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
