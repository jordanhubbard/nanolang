/* I observe the unchanged C implementation; I implement no graph operation. */
#include "../src/nanoisa/nvm2c_shape.c"
#include <inttypes.h>

static uint32_t word(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static int framed(const unsigned char *p, size_t n) {
    if(n<4)return 0;
    uint32_t count=word(p);size_t cursor=4;
    for(uint32_t item=0;item<count;item++) {
        if(n-cursor<8)return 0;
        uint32_t mode=word(p+cursor),ops=word(p+cursor+4);cursor+=8;
        if(mode>1 || ops>(n-cursor)/24)return 0;
        for(uint32_t i=0;i<ops;i++) {
            const unsigned char *at=p+cursor+(size_t)i*24;
            uint32_t op=word(at);
            if(op>13 || word(at+20)>1 || (op==6&&word(at+8)>65535) || (op==10&&word(at+12)>1))return 0;
        }
        cursor+=(size_t)ops*24;
    }
    return cursor==n;
}
static void event(const char *label,uint32_t item,uint32_t i,uint64_t result,uint32_t output,NvmShapeGraph *g) {
    printf("%s|%u|%u|%"PRIu64"|%u|%zu|%s\n",label,item,i,result,output,g->count,g->error?g->error:"");
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
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    FILE *file=fopen(argv[1],"rb");if(!file)return 2;
    if(fseek(file,0,SEEK_END)){fclose(file);return 2;}
    long end=ftell(file);if(end<0){fclose(file);return 2;}
    if((uint64_t)end>16777216){fclose(file);puts("INVALID");return 0;}
    rewind(file);size_t n=(size_t)end;unsigned char *p=malloc(n?n:1);if(!p){fclose(file);return 2;}
    if(fread(p,1,n,file)!=n){free(p);fclose(file);return 2;}
    if(fclose(file)){free(p);return 2;}
    if(!framed(p,n)){puts("INVALID");free(p);return 0;}
    for(int kind=0;kind<=NVM_SHAPE_BSTRING;kind++)printf("K|%d|%s\n",kind,kind_name((NvmShapeKind)kind));
    size_t cursor=4;uint32_t cases=word(p);
    for(uint32_t item=0;item<cases;item++) {
        uint32_t mode=word(p+cursor),ops=word(p+cursor+4);cursor+=8;NvmShapeGraph g={0};
        for(uint32_t i=0;i<ops;i++) {
            uint32_t op=word(p+cursor),a=word(p+cursor+4),b=word(p+cursor+8),c=word(p+cursor+12);
            uint32_t expected=word(p+cursor+16),check=word(p+cursor+20),output=305419896;uint64_t result=0;cursor+=24;
            switch(op) {
            case 0:result=nvm_shape_new(&g,(NvmShapeKind)a);break;
            case 1:result=nvm_shape_root(&g,a);break;
            case 2:result=nvm_shape_kind(&g,a);break;
            case 3:result=nvm_shape_child(&g,a,b);break;
            case 4:result=nvm_shape_lookup(&g,a,b);break;
            case 5:result=nvm_shape_unify(&g,a,b);break;
            case 6:result=nvm_shape_tags_add(&g,a,(uint16_t)b);break;
            case 7:result=nvm_shape_tags(&g,a);break;
            case 8:result=nvm_shape_function_add(&g,a,b);break;
            case 9:result=nvm_shape_function_count(&g,a);break;
            case 10:result=nvm_shape_function_target(&g,a,b,c?&output:NULL);break;
            case 11:nvm_shape_destroy(&g);break;
            case 12:result=nvm_shape_function_target(&g,a,((uint64_t)c<<32)|b,&output);break;
            case 13:result=!g.error;break;
            }
            if(check&&result!=expected){printf("ASSERT|%u|%u|%"PRIu64"|%u\n",item,i,result,expected);nvm_shape_destroy(&g);free(p);return 0;}
            if(!mode)event("E",item,i,result,output,&g);
        }
        event("Z",item,ops,0,305419896,&g);snapshot(&g);nvm_shape_destroy(&g);
    }
    free(p);return ferror(stdout)?3:0;
}
