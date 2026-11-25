/* I expose unchanged private analysis-call context initialization. */
#define main nvm2c_affine_frame_reference_main
#include "nvm2c_affine_frame_reference.c"
#undef main

static void context(Reader *r) {
    uint32_t entry=word(r),present=word(r),size=word(r);size_t at=0;NvmModule model={0};
    if(!r->ok || present>1 || !take(&r->pos,size,r->size,&at) || !load_model(r->bytes+at,size,&model)){r->ok=false;return;}
    if(r->pos==r->size) {
        model.header.entry_point=entry;AnalysisCalls *calls=calloc(1,sizeof(*calls));if(!calls)exit(2);
        analysis_calls_init(calls,present?&model:NULL);
        printf("%u,%u,%u,%u|",calls->globals_valid,calls->value_graph,calls->effect_globals,calls->global_count);
        for(uint32_t i=0;i<calls->global_count;i++){number(calls->globals[i].tag);number(calls->globals[i].mutable);number(calls->globals[i].layout);putchar(';');}
        free(calls);
    } else r->ok=false;
    free_model(&model);
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        FILE *file=fopen(argv[i],"rb");if(!file){puts("D");continue;}
        if(fseek(file,0,SEEK_END)){fclose(file);puts("D");continue;}
        long length=ftell(file);if(length<4 || fseek(file,0,SEEK_SET)){fclose(file);puts("D");continue;}
        uint8_t *bytes=malloc((size_t)length);if(!bytes)exit(2);
        bool loaded=fread(bytes,1,(size_t)length,file)==(size_t)length;fclose(file);Reader r={bytes,(size_t)length,0,loaded};uint32_t kind=word(&r);
        if(kind==0)flows(&r);else if(kind==1)queue(&r);else if(kind==2)frames(&r);else if(kind==3)profile(&r);
        else if(kind==14)context(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
