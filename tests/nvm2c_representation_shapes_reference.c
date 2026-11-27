/* I call unchanged logical adapters with an actual unchanged C shape graph. */
#include "../src/nanoisa/nvm2c_shape.c"
#include "../src/nanoisa/isa.h"
#include <stdarg.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char *err; size_t err_len; int failed;
    NvmShapeGraph shapes;
    int track_shapes,shape_generic_array;
    const void *map_plan;
    uint8_t shape_opcode;
    uint32_t classify_function_index;
    size_t classify_offset,record_width;
} Nvm2cBuf;
#include "nvm2c_representation_shapes_actual.h"
static uint32_t word(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static size_t short_word(const unsigned char *p) {return (size_t)p[0] | (size_t)p[1]<<8;}
static int framed(const unsigned char *p,size_t n) {
    if(n<4||n>1048576)return 0;
    uint32_t count=word(p);size_t cursor=4;
    if(count>(n-4)/16)return 0;
    for(uint32_t item=0;item<count;item++) {
        if(n-cursor<16)return 0;
        if(p[cursor]>1||p[cursor+1]>31||p[cursor+3])return 0;
        uint32_t ops=word(p+cursor+12);cursor+=16;
        if(ops>65535||ops>(n-cursor)/13)return 0;
        for(uint32_t i=0;i<ops;i++) {
            if(n-cursor<13)return 0;
            unsigned op=p[cursor];uint32_t a=word(p+cursor+1),b=word(p+cursor+5);
            if(op>20||(op==5&&b>65535)||(op==10&&b>14)||
               ((op==11||op==16||op==17)&&b>255)||(op==14&&(a>255||b>65535))||
               (op==15&&b>65535)||(op==20&&a>31))return 0;
            cursor+=13;
            if(op==18) {
                if(n-cursor<2)return 0;
                size_t width=short_word(p+cursor);cursor+=2;
                if(width>(n-cursor)/2)return 0;
                cursor+=width*2;
            }
        }
    }
    return cursor==n;
}
static void configure(Nvm2cBuf *b,unsigned flags) {
    static const char plan_present;
    b->track_shapes=(flags&1)!=0;b->shape_generic_array=(flags&2)!=0;
    b->map_plan=(flags&4)?&plan_present:NULL;b->failed=(flags&8)!=0;
    snprintf(b->err,b->err_len,"%s",(flags&16)?"prior":"");
}
static void event(const char *label,uint32_t item,uint32_t index,uint32_t result,Nvm2cBuf *b) {
    printf("%s|%u|%u|%u|%d|%s|%zu|%s\n",label,item,index,result,b->failed,b->err,b->shapes.count,b->shapes.error?b->shapes.error:"");
}
static void snapshot(NvmShapeGraph *g) {
    if(g->error)return;
    for(NvmShapeId id=1;id<=g->count;id++) {
        NvmShapeId root=nvm_shape_root(g,id);NvmShapeNode *node=&g->nodes[root-1];
        printf("N|%u|%u|%u|%u|%zu|",id,root,(unsigned)nvm_shape_kind(g,id),(unsigned)nvm_shape_tags(g,id),node->count);
        for(size_t i=0;i<node->count;i++)printf("%s%u:%u",i?",":"",node->edges[i].index,nvm_shape_root(g,node->edges[i].child));
        printf("|%zu|",node->function_count);
        for(size_t i=0;i<node->function_count;i++)printf("%s%u",i?",":"",node->functions[i]);
        putchar('\n');
    }
}
static void solver_snapshot(NvmShapeGraph *g) {
    snapshot(g);
    printf("D|%zu|%zu|%zu|%zu\n",g->conversion_count,g->array_alias_count,g->array_write_count,g->array_read_count);
    NvmShapeConversion *groups[]={g->conversions,g->array_aliases,g->array_writes};
    size_t counts[]={g->conversion_count,g->array_alias_count,g->array_write_count};
    const char *labels[]={"C","A","W"};
    for(size_t k=0;k<3;k++)for(size_t i=0;i<counts[k];i++) {
        NvmShapeConversion c=groups[k][i];printf("%s|%u|%u|%d|%d\n",labels[k],c.source,c.target,c.alias_view,c.checked_payload);
    }
    for(size_t i=0;i<g->array_read_count;i++){NvmShapeArrayRead r=g->array_reads[i];printf("R|%u|%u|%d|0\n",r.element,r.result,r.resolved);}
    if(!g->error)for(NvmShapeId id=1;id<=g->count;id++)printf("I|%u|%d\n",id,g->nodes[nvm_shape_root(g,id)-1].conversion_kind);
}
int main(int argc,char **argv) {
    for(int arg=1;arg<argc;arg++) {
        FILE *file=fopen(argv[arg],"rb");if(!file)return 2;
        struct stat st;
        if(fstat(fileno(file),&st)||!S_ISREG(st.st_mode)||st.st_size<0){fclose(file);return 2;}
        if((uint64_t)st.st_size>1048576){fclose(file);puts("INVALID");continue;}
        size_t n=(size_t)st.st_size;unsigned char *p=malloc(n?n:1);if(!p){fclose(file);return 2;}
        if(fread(p,1,n,file)!=n||ferror(file)){free(p);fclose(file);return 2;}
        if(fclose(file)){free(p);return 2;}
        if(!framed(p,n)){free(p);puts("INVALID");continue;}
        size_t cursor=4;uint32_t cases=word(p);
        for(uint32_t item=0;item<cases;item++) {
            unsigned mode=p[cursor],flags=p[cursor+1];char error[1024]="";
            Nvm2cBuf b={0};b.err=error;b.err_len=sizeof error;b.shape_opcode=p[cursor+2];
            b.classify_function_index=word(p+cursor+4);b.classify_offset=word(p+cursor+8);
            uint32_t ops=word(p+cursor+12);cursor+=16;configure(&b,flags);
            for(uint32_t i=0;i<ops;i++) {
                unsigned op=p[cursor];uint32_t a=word(p+cursor+1),v=word(p+cursor+5),c=word(p+cursor+9),result=0;cursor+=13;
                switch(op) {
                case 0:result=nvm_shape_new(&b.shapes,(NvmShapeKind)a);break;
                case 1:result=nvm_shape_child(&b.shapes,a,v);break;
                case 2:result=nvm_shape_unify(&b.shapes,a,v);break;
                case 3:result=nvm_shape_convert(&b.shapes,a,v);break;
                case 4:result=nvm_shape_solve_conversions(&b.shapes);break;
                case 5:result=nvm_shape_tags_add(&b.shapes,a,(uint16_t)v);break;
                case 6:result=nvm_shape_function_add(&b.shapes,a,v);break;
                case 7:nvm_shape_destroy(&b.shapes);break;
                case 8:result=shape_ok(&b);break;
                case 9:result=shape_variable(&b,&a);break;
                case 10:result=shape_type(&b,a,(NvmShapeKind)v);break;
                case 11:result=shape_kind(&b,a,(uint8_t)v);break;
                case 12:result=shape_equal(&b,a,v);break;
                case 13:result=shape_child(&b,a,v);break;
                case 14:{Nvm2cSimSlot slot={0};slot.kind=(uint8_t)a;slot.scalar_tags=(uint16_t)v;slot.shape=c;result=checked_array_payload(&b,&slot);break;}
                case 15:result=shape_carrier_box(&b,a,(uint16_t)v);break;
                case 16:result=shape_field_kind(&b,a,(uint8_t)v);break;
                case 17:result=private_map_shape(&b,a,(uint8_t)v);break;
                case 18:{size_t width=short_word(p+cursor);cursor+=2;b.record_width=width;result=shape_record_return(&b,a,v,p+cursor,p+cursor+width);cursor+=2*width;break;}
                case 19:result=resolved_shape_kind(&b,a);break;
                case 20:configure(&b,a);break;
                }
                if(!mode)event("E",item,i,result,&b);
            }
            event("Z",item,ops,0,&b);solver_snapshot(&b.shapes);nvm_shape_destroy(&b.shapes);
        }
        free(p);
    }
    return 0;
}
