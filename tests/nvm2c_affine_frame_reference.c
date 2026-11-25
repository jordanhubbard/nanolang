/* I expose the actual unchanged C frame/worklist helpers, not a second model. */
#include "../src/nanoisa/affine_bytecode.c"
#include "nvm2c_affine_model_reader.h"
#include <assert.h>

typedef struct { const uint8_t *bytes; size_t size, pos; bool ok; } Reader;
static uint32_t word(Reader *r) {
    size_t at=0;if(!r->ok || !take(&r->pos,4,r->size,&at)){r->ok=false;return 0;}
    return read32(r->bytes+at);
}
static void number(uint32_t n){printf("%u,",n);}
static void local_snapshot(const NvmAffineState *s,const NvmModule *m) {
    for(uint16_t i=0;i<m->functions[0].local_count;i++) {
        uint8_t tag=0,mode=0;uint16_t variant=UINT16_MAX;
        bool live=nvm_affine_local_info(s,i,&tag,&mode);
        (void)nvm_affine_union_variant(s,i,&variant);
        number(live);number(tag);number(mode);number(variant);
    }
}
static void snapshot(const Frame *f,const NvmModule *m) {
    local_snapshot(f->locals,m);putchar('/');
    for(uint16_t i=0;i<f->count;i++) {
        Value v=f->stack[i];number(v.tag);number(v.observation);number(v.root);
        number(v.owned);number(v.layout);number(v.variant);
        for(unsigned b=0;b<256;b++)putchar((v.excluded[b/64]>>(b%64))&1?'1':'0');
        putchar(';');
    }
    putchar('/');for(uint16_t i=0;i<f->global_count;i++)number(f->globals[i]);
    putchar('/');number(f->handler_floor);number(f->activation);
    for(uint16_t i=0;i<f->handler_count;i++){number(f->handlers[i].key);number(f->handlers[i].site);}
}
static Frame *read_frame(Reader *r,const NvmAffineState *initial) {
    uint32_t floor=word(r),activation=word(r),globals=word(r),handlers=word(r),stack=word(r),commands=word(r);
    if(!r->ok || globals>256 || handlers>64 || floor>handlers || stack>256 || commands>256){r->ok=false;return NULL;}
    Frame *f=calloc(1,sizeof(*f));if(!f)exit(2);
    f->locals=nvm_affine_state_clone(initial);if(!f->locals)exit(2);
    f->handler_floor=(uint16_t)floor;f->activation=activation;
    for(uint32_t i=0;r->ok && i<commands;i++) {
        uint32_t op=word(r),a=word(r),b=word(r),c=word(r);bool ok=false;NvmAffineType type;
        if(a>65535 || c>65535){r->ok=false;break;}
        switch(op) {
        case 0:ok=nvm_affine_scalar_define(f->locals,(uint16_t)a);break;
        case 1:ok=nvm_affine_string_define(f->locals,(uint16_t)a);break;
        case 2:ok=nvm_affine_record_define(f->locals,(uint16_t)a,b);break;
        case 3:ok=nvm_affine_union_define(f->locals,(uint16_t)a,b,(uint16_t)c);break;
        case 4:ok=nvm_affine_array_define(f->locals,(uint16_t)a,b);break;
        case 5:ok=nvm_affine_take_local(f->locals,(uint16_t)a,&type);break;
        case 6:ok=nvm_affine_region_begin(f->locals);break;
        case 7:ok=nvm_affine_region_end(f->locals);break;
        case 8:ok=nvm_affine_borrow(f->locals,a,(uint16_t)b,NULL,0,(NvmReferenceMode)c);break;
        }
        if(!ok)r->ok=false;
    }
    for(uint32_t i=0;r->ok && i<globals;i++){uint32_t g=word(r);if(g>15)r->ok=false;f->globals[f->global_count++]=(uint8_t)g;}
    for(uint32_t i=0;r->ok && i<handlers;i++){uint32_t key=word(r),site=word(r);f->handlers[f->handler_count++]=(Handler){key,site};}
    for(uint32_t i=0;r->ok && i<stack;i++) {
        uint32_t tag=word(r),observation=word(r),root=word(r),owned=word(r),layout=word(r),variant=word(r);
        if(tag>255 || observation>1 || root>65535 || owned>1 || (variant>=256 && variant!=65535)){r->ok=false;break;}
        Value v={.tag=(uint8_t)tag,.observation=observation!=0,.root=(uint16_t)root,
            .owned=owned!=0,.layout=layout,.variant=(uint16_t)variant};
        size_t at=0;if(!take(&r->pos,256,r->size,&at)){r->ok=false;break;}
        for(unsigned b=0;b<256;b++){uint8_t flag=r->bytes[at+b];if(flag>1)r->ok=false;if(flag)v.excluded[b/64]|=UINT64_C(1)<<(b%64);}
        if(!push(f,v))exit(2);
    }
    return f;
}
static void frames(Reader *r) {
    uint32_t operation=word(r),argument=word(r),size=word(r);size_t at=0;
    NvmModule model={0};
    if(!r->ok || operation>6 || argument>65535 || !take(&r->pos,size,r->size,&at) ||
       !load_model(r->bytes+at,size,&model) || !model.function_count){r->ok=false;return;}
    NvmAffineState *initial=nvm_affine_state_create(&model,0,model.functions[0].local_count);
    if(!initial){free_model(&model);r->ok=false;return;}
    Frame *a=read_frame(r,initial),*b=read_frame(r,initial),*out=NULL;
    nvm_affine_state_free(initial);
    bool ok=false,changed=false;const char *error="";
    if(r->ok && ((operation==5 && b->count!=1) || (operation==6 && argument>255)))r->ok=false;
    if(r->ok && r->pos==r->size) {
        out=frame_clone(a);if(!out)exit(2);
        if(operation==0)ok=stack_meet(out,b,&changed);
        if(operation==1) {
            Frame *nodes[]={out};uint32_t items[1]={0};bool queued[1]={false};
            Worklist work={.items=items,.queued=queued,.capacity=1};
            ok=propagate(nodes,0,b,&work,&error);out=nodes[0];changed=work.pending!=0;
        }
        if(operation==2)ok=argument<256 && match_refine(out,(uint16_t)argument);
        if(operation==3)ok=true;
        if(operation==4)ok=observes_owner(out,(uint16_t)argument);
        if(operation==5)changed=ok=push(out,b->stack[0]);
        if(operation==6)changed=ok=pop_scalar(out,(uint8_t)argument);
        if(!ok){frame_free(out);out=frame_clone(a);changed=false;}
        if(operation!=1)error="";
        printf("%u%u|%s|",ok,changed,error);snapshot(out,&model);
    } else r->ok=false;
    frame_free(a);frame_free(b);frame_free(out);free_model(&model);
}
static void queue(Reader *r) {
    uint32_t capacity=word(r),count=word(r);
    if(!r->ok || !capacity || capacity>4096 || count>100000){r->ok=false;return;}
    Worklist w={.capacity=capacity};w.items=calloc(capacity,sizeof(*w.items));w.queued=calloc(capacity,sizeof(*w.queued));
    if(!w.items || !w.queued)exit(2);
    for(uint32_t i=0;r->ok && i<count;i++) {
        uint32_t op=word(r),index=word(r);bool ok=false;int64_t result=-1;
        if(op>1 || (op==0 && index>=capacity)){r->ok=false;break;}
        if(op==0){ok=enqueue(&w,index);result=index;}
        else if(w.pending){ok=true;result=w.items[w.head];w.head=(w.head+1)%capacity;w.pending--;w.queued[result]=false;}
        printf("%u,%lld,%u,%u,%u:",ok,(long long)result,w.head,w.tail,w.pending);
        if(w.pending)number(w.items[w.head]);
        putchar(';');
    }
    free(w.items);free(w.queued);
}
static void flows(Reader *r) {
    uint32_t count=word(r);
    if(count>200000){r->ok=false;return;}
    for(uint32_t i=0;r->ok && i<count;i++) {
        uint32_t op=word(r),a=word(r),b=word(r),c=word(r),d=word(r),e=word(r),f=word(r);
        if(op>4 || a>255 || b>255 || c>255 || d>255 || e>255 || f>255){r->ok=false;break;}
        if(op==0){number(nvm_affine_global_require(a,b));putchar(';');continue;}
        if(op==1){number(nvm_affine_global_compose(a,b));putchar(';');continue;}
        NvmAffineGlobalFlow first=nvm_affine_global_flow(a,b,c),second=nvm_affine_global_flow(d,e,f),result=first;
        if(op==3)result=nvm_affine_global_then(first,second);
        if(op==4)result=nvm_affine_global_join(first,second);
        number(result.allowed);number(result.normal);number(result.lexical_return);putchar(';');
    }
}
static void profile(Reader *r) {
    uint32_t opcode=word(r),graph=word(r);
    if(!r->ok || opcode>255 || graph>1 || r->pos!=r->size){r->ok=false;return;}
    putchar(supported((uint8_t)opcode,graph!=0)?'1':'0');
}
int main(int argc,char **argv) {
    for(int i=1;i<argc;i++) {
        FILE *file=fopen(argv[i],"rb");if(!file){puts("D");continue;}
        if(fseek(file,0,SEEK_END)){fclose(file);puts("D");continue;}
        long length=ftell(file);if(length<4 || fseek(file,0,SEEK_SET)){fclose(file);puts("D");continue;}
        uint8_t *bytes=malloc((size_t)length);if(!bytes)exit(2);
        bool loaded=fread(bytes,1,(size_t)length,file)==(size_t)length;fclose(file);
        Reader r={bytes,(size_t)length,0,loaded};uint32_t kind=word(&r);
        if(kind==0)flows(&r);else if(kind==1)queue(&r);else if(kind==2)frames(&r);else if(kind==3)profile(&r);else r.ok=false;
        if(!r.ok || r.pos!=r.size)printf("D");
        putchar('\n');free(bytes);
    }
    return 0;
}
