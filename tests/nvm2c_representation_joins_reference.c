/* I execute verbatim classifier helpers; my owner models only fields they read. */
#include "../src/nanoisa/nvm2c_shape.c"
#include "../src/nanoisa/isa.h"
#include <stdarg.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <unistd.h>
_Static_assert(sizeof(int)==4,"I require the actual 32-bit C origin contract");
#include "nvm2c_representation_joins_types.h"
typedef struct {
    Nvm2cScalarJoin *scalar_joins;
    Nvm2cFieldBlock *field_blocks;
    int has_maps;
    char *err; size_t err_len; int failed;
    NvmShapeGraph shapes;
    int track_shapes,shape_generic_array;
    uint8_t shape_opcode;
    uint32_t classify_function_index;
    size_t classify_offset,record_width,sim_stack_capacity;
    uint8_t *default_fields;
    NvmShapeId *shape_current;
} Nvm2cBuf;
#include "nvm2c_representation_joins_actual.h"
static uint32_t word(const unsigned char *p) {
    return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
}
static unsigned short_word(const unsigned char *p) {return (unsigned)p[0]|(unsigned)p[1]<<8;}
static int origin(const unsigned char *p) {uint32_t bits=word(p);int32_t value;memcpy(&value,&bits,4);return value;}
static int framed(const unsigned char *p,size_t n) {
    if(n<4||n>4194304)return 0;
    uint32_t count=word(p);size_t at=4;
    if(count>(n-4)/36)return 0;
    for(uint32_t item=0;item<count;item++) {
        if(n-at<36||p[at+2]>1||p[at+3])return 0;
        uint32_t low=word(p+at+16),high=word(p+at+20),ops=word(p+at+32);
        unsigned width=short_word(p+at+28),locals=short_word(p+at+30);
        if(high>1||(high&&low)||ops>131072)return 0;
        at+=36;if((size_t)width+locals>n-at)return 0;at+=(size_t)width+locals;
        if(ops>(n-at)/20)return 0;
        for(uint32_t i=0;i<ops;i++) {
            if(n-at<20)return 0;
            unsigned op=p[at],present=p[at+2];uint32_t parameter=word(p+at+16);int value=origin(p+at+4);
            if(op>26||present>1||short_word(p+at+14)||(op==8&&parameter>255)||(op==9&&parameter)||(op==11&&parameter>1))return 0;
            if((op==12||op==13)&&(value<0||value>131072))return 0;
            if(op==16&&p[at+1]>14)return 0;
            /* This is a debug-transport storage precondition, not a product refusal. */
            if((op==3||op==4)&&value>=0&&(uint16_t)value<locals&&(uint32_t)value>=locals)return 0;
            at+=20;if(present){if(width>n-at)return 0;at+=width;}
        }
    }
    return at==n;
}
static Nvm2cSimSlot read_slot(unsigned char *p) {
    Nvm2cSimSlot value={0};value.kind=p[1];value.origin=origin(p+4);
    value.rec_k=p[2]?p+20:NULL;value.array_payload_kind=p[3];
    value.shape=word(p+8);value.scalar_tags=(uint16_t)short_word(p+12);return value;
}
static Nvm2cSimSlot no_output(void) {Nvm2cSimSlot v={0};v.kind=NVM2C_VK_UNK;v.origin=-1;return v;}
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

static void slot_line(const char *label,int index,Nvm2cSimSlot value,size_t width) {
    printf("%s|%d|%u|%d|%d|%u|%u|%u|",label,index,value.kind,value.origin,value.rec_k!=NULL,value.shape,value.scalar_tags,value.array_payload_kind);
    if(value.rec_k)for(size_t i=0;i<width;i++)printf("%s%u",i?",":"",value.rec_k[i]);
    putchar('\n');
}
static void event(const char *label,uint32_t item,uint32_t index,int accepted,Nvm2cBuf *b,int sp,NvmShapeId current) {
    printf("%s|%u|%u|%d|%d|%s|%d|%u|%zu|%s\n",label,item,index,accepted,b->failed,b->err,sp,current,b->shapes.count,b->shapes.error?b->shapes.error:"");
}
static void complete(Nvm2cBuf *b,Nvm2cSimSlot *cells,int sp,uint8_t *locals,size_t count,Nvm2cSimSlot output) {
    printf("L|");for(size_t i=0;i<count;i++)printf("%s%u",i?",":"",locals[i]);putchar('\n');
    slot_line("V",-1,output,b->record_width);
    for(int i=0;i<sp;i++)slot_line("S",i,cells[i],b->record_width);
    solver_snapshot(&b->shapes);
}
static void join_report(Nvm2cBuf *b,Nvm2cFacts *facts,Nvm2cSimJoin *join,int allocated) {
    printf("J|%d|%d|%d|%d|%d|%d\n",join->set,join->sp,allocated,facts->changed,facts->final,b->has_maps);
    for(int i=0;i<allocated;i++)slot_line("T",i,join->slots[i],b->record_width);
    int index=0;
    for(Nvm2cScalarJoin *p=b->scalar_joins;p;p=p->next,index++) {
        printf("P|%d|%u|%zu|%d|",index,p->function,p->target,p->count);
        for(int i=0;i<p->count;i++)printf("%s%u",i?",":"",p->tags[i]);
        putchar('\n');
    }
}
static int plan_metadata(Nvm2cBuf *b,Nvm2cSimSlot value,uint32_t target,int create) {
    if(create) {
        size_t count=(size_t)value.origin;
        Nvm2cScalarJoin *p=calloc(1,sizeof *p+count*sizeof *p->tags);
        if(!p)return 0;
        p->function=value.shape;p->target=target;p->count=value.origin;
        for(int i=0;i<p->count;i++)p->tags[i]=value.scalar_tags;
        p->next=b->scalar_joins;b->scalar_joins=p;
    } else {
        Nvm2cScalarJoin *p=b->scalar_joins;
        while(p&&(p->function!=value.shape||p->target!=target))p=p->next;
        if(p&&value.origin<p->count)p->tags[value.origin]=value.scalar_tags;
    }
    return 1;
}

int main(int argc,char **argv) {
    for(int arg=1;arg<argc;arg++) {
        FILE *file=fopen(argv[arg],"rb");if(!file)return 2;struct stat st;
        if(fstat(fileno(file),&st)||!S_ISREG(st.st_mode)||st.st_size<0){fclose(file);return 2;}
        if((uint64_t)st.st_size>4194304){fclose(file);puts("INVALID");continue;}
        size_t n=(size_t)st.st_size;unsigned char *data=malloc(n?n:1);if(!data){fclose(file);return 2;}
        if(fread(data,1,n,file)!=n||ferror(file)){free(data);fclose(file);return 2;}
        if(fclose(file)){free(data);return 2;}
        if(!framed(data,n)){free(data);puts("INVALID");continue;}
        size_t at=4;uint32_t count=word(data);
        for(uint32_t item=0;item<count;item++) {
            unsigned flags=data[at],mode=data[at+2];char error[1024];Nvm2cBuf b={0};NvmShapeId current=word(data+at+24);
            b.err=error;b.err_len=sizeof error;b.failed=(flags&8)!=0;b.track_shapes=(flags&1)!=0;b.shape_generic_array=(flags&2)!=0;
            b.shape_opcode=data[at+1];b.classify_function_index=word(data+at+4);b.classify_offset=word(data+at+8);
            uint32_t function=word(data+at+12),ops=word(data+at+32);
            b.sim_stack_capacity=(size_t)((uint64_t)word(data+at+16)+((uint64_t)word(data+at+20)<<32));b.record_width=short_word(data+at+28);
            uint16_t locals_count=(uint16_t)short_word(data+at+30);b.shape_current=&current;
            snprintf(error,sizeof error,"%s",(flags&16)?"prior":"");
            at+=36;b.default_fields=data+at;at+=b.record_width;uint8_t *locals=data+at;at+=locals_count;
            Nvm2cSimSlot *cells=calloc((size_t)ops+1,sizeof *cells),*saved=calloc((size_t)ops+1,sizeof *saved);
            if(!cells||!saved){free(cells);free(saved);nvm_shape_destroy(&b.shapes);free(data);return 2;}
            int sp=0,saved_sp=0;Nvm2cSimSlot output=no_output();
            Nvm2cFacts facts={0};facts.changed=(flags&64)!=0;facts.final=(flags&32)!=0;b.has_maps=(flags&128)!=0;
            Nvm2cSimJoin join={0};int allocated=0;size_t retired_count=0;
            Nvm2cSimSlot **join_storage=calloc((size_t)ops+1,sizeof *join_storage);
            if(!join_storage)return 2;
            for(uint32_t index=0;index<ops;index++) {
                unsigned op=data[at];uint32_t parameter=word(data+at+16);Nvm2cSimSlot value=read_slot(data+at);at+=20+(value.rec_k?b.record_width:0);
                int accepted=1;output=no_output();
                switch(op) {
                case 0:accepted=sim_push_slot(&b,function,cells,&sp,value);break;
                case 1:accepted=sim_push(&b,function,cells,&sp,value.kind,value.origin);break;
                case 2:output=value;accepted=sim_pop(&b,function,cells,&sp,&output);break;
                case 3:accepted=mark_string_operand(&b,locals,locals_count,value);break;
                case 4:accepted=require_typed_integer_operand(&b,locals,locals_count,value);break;
                case 5:current=parameter;break;
                case 6:saved_sp=sp;if(sp)memcpy(saved,cells,(size_t)sp*sizeof *cells);break;
                case 7:sp=saved_sp;if(sp)memcpy(cells,saved,(size_t)sp*sizeof *cells);break;
                case 8:b.shape_opcode=(uint8_t)parameter;break;
                case 9:accepted=nvm_shape_solve_conversions(&b.shapes);break;
                case 10: {
                    for(int i=0;i<sp;i++)if(b.record_width&&!cells[i].rec_k&&(cells[i].kind==4||cells[i].kind==6||cells[i].kind==7)&&join.set)return 2;
                    Nvm2cSimSlot *old=join.slots;
                    accepted=sim_join(&b,function,parameter,&join,cells,sp,&facts);
                    if(join.slots!=old) {
                        allocated=join.slots?sp:0;
                        if(join.slots)join_storage[retired_count++]=join.slots;
                    }
                    break;
                }
                case 11:facts.final=parameter!=0;break;
                case 12:case 13:if(!plan_metadata(&b,value,parameter,op==12))return 2;break;
                case 14:function=parameter;break;
                case 15:if((size_t)sp>=b.sim_stack_capacity)return 2;cells[sp++]=value;break;
                case 16:output.shape=nvm_shape_new(&b.shapes,(NvmShapeKind)value.kind);accepted=output.shape!=0;break;
                case 17:output.shape=nvm_shape_child(&b.shapes,value.shape,parameter);accepted=output.shape!=0;break;
                case 18:accepted=nvm_shape_unify(&b.shapes,value.shape,parameter);break;
                case 19:accepted=nvm_shape_convert(&b.shapes,value.shape,parameter);break;
                case 20:accepted=nvm_shape_tags_add(&b.shapes,value.shape,value.scalar_tags);break;
                case 21:accepted=nvm_shape_array_write(&b.shapes,value.shape,parameter);break;
                case 22:accepted=nvm_shape_array_read(&b.shapes,value.shape,parameter);break;
                case 23:accepted=nvm_shape_function_add(&b.shapes,value.shape,parameter);break;
                case 24:accepted=nvm_shape_array_alias(&b.shapes,value.shape,parameter);break;
                case 25:accepted=nvm_shape_checked_payload(&b.shapes,value.shape,parameter);break;
                case 26:accepted=nvm_shape_alias_view(&b.shapes,value.shape,parameter);break;
                }
                event("E",item,index,accepted,&b,sp,current);
                if(!mode){complete(&b,cells,sp,locals,locals_count,output);join_report(&b,&facts,&join,allocated);}
            }
            event("Z",item,ops,0,&b,sp,current);if(mode||!ops){complete(&b,cells,sp,locals,locals_count,output);join_report(&b,&facts,&join,allocated);}
            for(size_t i=0;i<retired_count;i++)free(join_storage[i]);
            free(join_storage);
            while(b.scalar_joins){Nvm2cScalarJoin *next=b.scalar_joins->next;free(b.scalar_joins);b.scalar_joins=next;}
            while(b.field_blocks){Nvm2cFieldBlock *next=b.field_blocks->next;free(b.field_blocks);b.field_blocks=next;}
            free(cells);free(saved);nvm_shape_destroy(&b.shapes);
        }
        free(data);
    }
    return 0;
}
