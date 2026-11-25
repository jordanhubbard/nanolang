/* I compare complete owner/reference transitions with the unchanged C step. */
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main

void nvm2c_affine_owner_state_snapshot(const NvmAffineState *s);
bool nvm2c_affine_owner_state_region(NvmAffineState *s,uint32_t region);
typedef struct {uint32_t opcode,a,b,c;} OwnerStep;
static bool owner_handles(uint32_t op) {return (op>=11 && op<=14) || (op>=22 && op<=31);}
static void owner_transfer(Reader *r) {
    uint32_t count=word(r),override=word(r),region=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    if(!r->ok || !count || count>64 || override>1 || !take(&r->pos,size,r->size,&at) ||
       !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(!model.function_count || model.functions[0].local_count>NVM_AFFINE_MAX_LOCALS){free_model(&model);r->ok=false;return;}
    NvmAffineState *initial=nvm_affine_state_create(&model,0,model.functions[0].local_count);
    if(!initial){free_model(&model);r->ok=false;return;}
    Frame *current=read_frame(r,initial);nvm_affine_state_free(initial);
    if(r->ok && override && !nvm2c_affine_owner_state_region(current->locals,region))r->ok=false;
    OwnerStep *steps=calloc(count,sizeof(*steps));if(!steps)exit(2);
    for(uint32_t i=0;r->ok && i<count;i++) {
        steps[i].opcode=word(r);steps[i].a=word(r);steps[i].b=word(r);steps[i].c=word(r);
        if(!owner_handles(steps[i].opcode) || (steps[i].opcode!=OP_OWN_PACK && steps[i].a>UINT16_MAX) || steps[i].b>UINT16_MAX)r->ok=false;
    }
    if(r->ok && r->pos==r->size) {
        AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);
        for(uint32_t i=0;i<count;i++) {
            Frame *before=frame_clone(current),*out=frame_clone(current);if(!before || !out)exit(2);
            OwnerStep s=steps[i];DecodedInstruction instruction={0};instruction.opcode=(uint8_t)s.opcode;
            if(s.opcode==OP_OWN_PACK)instruction.operands[0].u32=s.a;
            else instruction.operands[0].u16=(uint16_t)s.a;
            instruction.operands[1].u16=(uint16_t)s.b;instruction.operands[2].u32=s.c;
            const char *error=step(out,&instruction,model.functions[0].local_count,&model,0,0,calls);
            if(error){frame_free(out);out=frame_clone(current);if(!out)exit(2);}
            assert(nvm_affine_state_equal(current->locals,before->locals));
            assert(current->count==before->count && current->global_count==before->global_count);
            assert(!current->count || !memcmp(current->stack,before->stack,current->count*sizeof(*current->stack)));
            if(i){putchar('~');}printf("%u|%s|",error==NULL,error?error:"");snapshot(out,&model);putchar('|');
            nvm2c_affine_owner_state_snapshot(out->locals);
            frame_free(before);frame_free(current);current=out;
        }
        free(calls);
    } else r->ok=false;
    free(steps);frame_free(current);free_model(&model);
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
        else if(kind==9)owner_transfer(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
