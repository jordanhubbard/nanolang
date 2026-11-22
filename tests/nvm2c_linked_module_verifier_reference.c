/* I compare public linked admission, then each present graph node, in order. */
#define main ordinary_module_reference_main
#include "nvm2c_module_verifier_reference.c"
#undef main

static uint32_t word(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
typedef struct { NvmV2Module wire; NvmModule *module; uint16_t *declared; } Loaded;
static NvmV2Result load(const unsigned char *bytes, size_t size, Loaded *out,
                        NvmVerifyResult *proof) {
    NvmV2Result status = nvm_v2_module_deserialize(bytes, size, &out->wire);
    if (status == NVM_V2_OK) status = nvm_v2_to_nvm_module(&out->wire, &out->module);
    if (status != NVM_V2_OK) return status;
    uint32_t count = out->module->function_count;
    out->declared = calloc(count ? count : 1, sizeof *out->declared);
    if (!out->declared || out->wire.functions.count != count) exit(2);
    for (uint32_t i=0; i<count; ++i) out->declared[i] = out->wire.functions.items[i].max_stack;
    *proof = nvm_verify_declared_max_stacks(out->module, out->declared, count);
    return status;
}
int main(int argc, char **argv) {
    bool graph = argc>1 && !strcmp(argv[1], "--graph");
    for (int arg=graph?2:1; arg<argc; ++arg) {
        FILE *file=fopen(argv[arg],"rb");
        if (!file || fseek(file,0,SEEK_END)) return 2;
        long length=ftell(file);
        if (length<0 || fseek(file,0,SEEK_SET)) return 2;
        size_t size=(size_t)length;
        unsigned char *bytes=malloc(size?size:1);
        if (!bytes || fread(bytes,1,size,file)!=size || fclose(file)) return 2;
        uint32_t root_size=0,count=0;
        bool envelope=size>=8;
        if (envelope) { root_size=word(bytes); count=word(bytes+4); envelope=count<=64 && root_size<=size-8; }
        Loaded nodes[65]={0}; const NvmModule *links[64]={0};
        NvmV2Result status=NVM_V2_OK; NvmVerifyResult proof={.ok=true};
        int failed=0; bool declarations=false;
        size_t cursor=8;
        if (envelope) {
            /* I validate the complete test carrier before calling product APIs. */
            cursor+=root_size;
            for (uint32_t i=0; envelope && i<count; ++i) {
                if (cursor>size || size-cursor<8) { envelope=false; break; }
                uint32_t resolved=word(bytes+cursor), span=word(bytes+cursor+4); cursor+=8;
                if (resolved>1 || span>size-cursor) { envelope=false; break; }
                cursor+=span;
            }
            envelope &= cursor==size;
        }
        if (!envelope) puts("X");
        else {
            status=load(bytes+8,root_size,&nodes[0],&proof);
            declarations=!proof.ok;
            cursor=8+root_size;
            for (uint32_t i=0; status==NVM_V2_OK && proof.ok && i<count; ++i) {
                uint32_t resolved=word(bytes+cursor),span=word(bytes+cursor+4); cursor+=8;
                if (resolved) {
                    status=load(bytes+cursor,span,&nodes[i+1],&proof);
                    if (status!=NVM_V2_OK || !proof.ok) { failed=(int)i+1; declarations=!proof.ok; }
                    links[i]=nodes[i+1].module;
                }
                cursor+=span;
            }
            if (status==NVM_V2_OK && proof.ok) {
                proof=nvm_verify_linked(nodes[0].module,links,count);
                if (proof.ok && graph) for (uint32_t i=0; i<count; ++i) if (links[i]) {
                    proof=nvm_verify_linked(links[i],links,count);
                    if (!proof.ok) { failed=(int)i+1; break; }
                }
            }
            if (graph) printf("%d|",status==NVM_V2_OK && proof.ok ? -1 : failed);
            if (status!=NVM_V2_OK) printf("R:%u\n",(unsigned)status);
            else if (!proof.ok) {
                fprintf(stderr,"%s\t%d\t%s\n",argv[arg],failed,proof.error_msg);
                refusal(nodes[failed].module,nodes[failed].declared,declarations,proof);
            } else {
                printf("1:1:");
                for (uint32_t i=0; i<nodes[0].module->function_count; ++i) {
                    uint16_t maximum=0xabcd;
                    NvmVerifyResult one=nvm_verify_function_max_stack(nodes[0].module,i,&maximum);
                    if (!one.ok) return 3;
                    printf("%s%u",i?",":"",maximum);
                }
                putchar('\n');
            }
        }
        for (unsigned i=0;i<65;++i) {
            free(nodes[i].declared); nvm_module_free(nodes[i].module); nvm_v2_module_free(&nodes[i].wire);
        }
        free(bytes);
    }
    return 0;
}
