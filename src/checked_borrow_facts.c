#include "checked_borrow_facts.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define BORROW_EVENT_LIMIT 4096u
#define BORROW_BYTE_LIMIT 1048576u
#define BORROW_WORK_LIMIT 131072u
#define BORROW_PATH_LIMIT 4096u

typedef struct { CheckedBorrowEvent event; char *bytes; } OwnedBorrowEvent;
typedef struct { CheckedFieldReadEvent event; char *bytes; } OwnedFieldRead;
typedef struct { CheckedFieldWriteEvent event; char *bytes; } OwnedFieldWrite;
struct CheckedBorrowLedger {
    Environment *owner;
    OwnedBorrowEvent *events;
    size_t count, capacity, total_events, total_bytes, work;
    OwnedFieldRead *reads;
    size_t read_count, read_capacity;
    OwnedFieldWrite *writes;
    size_t write_count, write_capacity;
    bool capturing, sealed, disabled, failed;
};
static void clear_events(CheckedBorrowLedger *s) {
    for(size_t i=0;i<s->count;i++) free(s->events[i].bytes);
    free(s->events);s->events=NULL;s->count=s->capacity=0;
    for(size_t i=0;i<s->read_count;i++)free(s->reads[i].bytes);
    free(s->reads);s->reads=NULL;s->read_count=s->read_capacity=0;
    for(size_t i=0;i<s->write_count;i++)free(s->writes[i].bytes);
    free(s->writes);s->writes=NULL;s->write_count=s->write_capacity=0;
}
static void disable(CheckedBorrowLedger *s) {
    clear_events(s);s->disabled=true;
}
static bool charge(CheckedBorrowLedger *s,size_t n) {
    if(n>BORROW_WORK_LIMIT-s->work){disable(s);return false;}
    s->work+=n;return true;
}
static bool utf8(const char *p,size_t n) {
    size_t i=0;
    while(i<n) {
        unsigned c=(unsigned char)p[i++],more=0;uint32_t value=0,min=0;
        if(c<128){if(!c)return false;continue;}
        if(c>=0xc2 && c<=0xdf){more=1;value=c&31;min=128;}
        else if(c>=0xe0 && c<=0xef){more=2;value=c&15;min=2048;}
        else if(c>=0xf0 && c<=0xf4){more=3;value=c&7;min=65536;}
        else return false;
        if(more>n-i)return false;
        while(more--){unsigned d=(unsigned char)p[i++];if((d&192)!=128)return false;value=(value<<6)|(d&63);}
        if(value<min || value>0x10ffff || (value>=0xd800 && value<=0xdfff))return false;
    }
    return true;
}
/* I bound scans themselves, not only the allocation that follows. */
static bool length(CheckedBorrowLedger *s,const char *p,size_t *out) {
    if(!p)return false;
    size_t n=0;
    while(p[n]){if(!charge(s,1))return false;n++;}
    if(!charge(s,1))return false;
    *out=n;return true;
}
bool checked_borrow_begin(Environment *env) {
    if(!env || env->checked_borrow_facts)return false;
    CheckedBorrowLedger *s=calloc(1,sizeof(*s));
    if(!s)return false;
    s->owner=env;s->capturing=true;env->checked_borrow_facts=s;return true;
}
void checked_borrow_finish_capture(Environment *env) {
    if(env && env->checked_borrow_facts)env->checked_borrow_facts->capturing=false;
}
bool checked_borrow_seal(Environment *env) {
    CheckedBorrowLedger *s=env?env->checked_borrow_facts:NULL;
    if(!s || s->owner!=env || s->capturing || s->failed)return false;
    s->sealed=true;return true;
}
bool checked_borrow_is_sealed(const Environment *env) {
    const CheckedBorrowLedger *s=env?env->checked_borrow_facts:NULL;
    return s && s->owner==env && s->sealed && !s->capturing && !s->failed;
}
void checked_borrow_end(Environment *env) {
    if(!env)return;
    CheckedBorrowLedger *s=env->checked_borrow_facts;env->checked_borrow_facts=NULL;
    if(s){clear_events(s);free(s);}
}
void checked_borrow_function_begin(Environment *env,ASTNode *function) {
    CheckedBorrowLedger *s=env?env->checked_borrow_facts:NULL;
    if(!s || !s->capturing || s->disabled || s->failed)return;
    for(size_t i=0;i<s->count;) {
        if(!charge(s,1))return;
        if(s->events[i].event.function==function){free(s->events[i].bytes);s->events[i]=s->events[--s->count];}
        else i++;
    }
    for(size_t i=0;i<s->read_count;) {
        if(!charge(s,1))return;
        if(s->reads[i].event.function==function){free(s->reads[i].bytes);s->reads[i]=s->reads[--s->read_count];}
        else i++;
    }
    for(size_t i=0;i<s->write_count;) {
        if(!charge(s,1))return;
        if(s->writes[i].event.function==function){free(s->writes[i].bytes);s->writes[i]=s->writes[--s->write_count];}
        else i++;
    }
}
static bool callee(const ASTNode *call,const ASTNode **expression,const char **name,const char **owner) {
    *expression=NULL;*owner="";
    if(call->type==AST_MODULE_QUALIFIED_CALL) {
        *name=call->as.module_qualified_call.function_name;*owner=call->as.module_qualified_call.module_alias;
        return *name && *owner;
    }
    if(call->type!=AST_CALL)return false;
    *expression=call->as.call.func_expr;
    if(*expression) {
        if((*expression)->type!=AST_IDENTIFIER)return false;
        *owner=(*expression)->as.identifier;
    }
    *name=call->as.call.name?call->as.call.name:"";
    return **name || **owner;
}
bool checked_borrow_capture(Environment *env,ASTNode *function,ASTNode *call,
    ASTNode *argument,unsigned ordinal,const char *root,const char *projection) {
    CheckedBorrowLedger *s=env?env->checked_borrow_facts:NULL;
    if(!s || !s->capturing || s->disabled)return true;
    if(s->failed)return false;
    if(!function || function->type!=AST_FUNCTION || !call || !argument ||
       argument->type!=AST_CALL || argument->as.call.arg_count!=1 ||
       (argument->as.call.borrow_mode!=1 && argument->as.call.borrow_mode!=2) ||
       call->line<=0 || call->column<=0 || argument->line<=0 || argument->column<=0 ||
       !function->as.function.affine_check_epoch || function->as.function.affine_check_epoch==UINT64_MAX)return true;
    if(!charge(s,1))return true;
    const ASTNode *callee_expression;const char *callee_name,*callee_owner;
    if(!callee(call,&callee_expression,&callee_name,&callee_owner))return true;
    const char *path=env->current_file;size_t np,nr,nj,nc,no;
    if(!length(s,path,&np) || !length(s,root,&nr) || !length(s,projection,&nj) ||
       !length(s,callee_name,&nc) || !length(s,callee_owner,&no))return true;
    if(!charge(s,np+nr+nj))return true;
    if(!np || !nr || !utf8(path,np) || !utf8(root,nr) || !utf8(projection,nj))return true;
    if(nr>BORROW_PATH_LIMIT || nj>BORROW_PATH_LIMIT-nr ||
       nc>BORROW_PATH_LIMIT || no>BORROW_PATH_LIMIT-nc ||
       np>BORROW_BYTE_LIMIT-s->total_bytes || nr>BORROW_BYTE_LIMIT-s->total_bytes-np ||
       nj>BORROW_BYTE_LIMIT-s->total_bytes-np-nr || s->total_events==BORROW_EVENT_LIMIT){disable(s);return true;}
    if(!charge(s,nj))return true;
    unsigned depth=0;for(size_t i=0;i<nj;i++)if(projection[i]=='.')depth++;
    if(depth>32){disable(s);return true;}
    for(size_t i=0;i<s->count;i++) {
        if(!charge(s,1))return true;
        if(s->events[i].event.function==function && s->events[i].event.argument==argument &&
           s->events[i].event.call==call && s->events[i].event.ordinal==ordinal)return true;
    }
    if(s->count==s->capacity) {
        size_t cap=s->capacity?s->capacity*2:16;
        if(cap>BORROW_EVENT_LIMIT)cap=BORROW_EVENT_LIMIT;
        if(cap>SIZE_MAX/sizeof(*s->events)){s->failed=true;return false;}
        void *next=realloc(s->events,cap*sizeof(*s->events));
        if(!next){s->failed=true;return false;}s->events=next;s->capacity=cap;
    }
    size_t size=np+nr+nj+nc+no+5;
    if(size>BORROW_BYTE_LIMIT-s->total_bytes){disable(s);return true;}
    char *bytes=malloc(size);
    if(!bytes){s->failed=true;return false;}
    memcpy(bytes,path,np+1);memcpy(bytes+np+1,root,nr+1);memcpy(bytes+np+nr+2,projection,nj+1);
    char *cn=bytes+np+nr+nj+3,*co=cn+nc+1;
    memcpy(cn,callee_name,nc+1);memcpy(co,callee_owner,no+1);
    CheckedBorrowEvent e={.function=function,.call=call,.argument=argument,.place=argument->as.call.args[0],
        .callee_expression=callee_expression,.callee_kind=call->type,.epoch=function->as.function.affine_check_epoch,
        .ordinal=ordinal,.call_line=(uint32_t)call->line,.call_column=(uint32_t)call->column,
        .argument_line=(uint32_t)argument->line,.argument_column=(uint32_t)argument->column,
        .mode=(unsigned)argument->as.call.borrow_mode,.source_path=bytes,.root=bytes+np+1,
        .projection=bytes+np+nr+2,.callee_name=cn,.callee_owner=co};
    s->events[s->count++]=(OwnedBorrowEvent){e,bytes};s->total_events++;s->total_bytes+=size;return true;
}
static bool same_text(const char *a,const char *b,size_t *budget) {
    if(!a || !b)return false;
    while(*budget){(*budget)--;if(*a!=*b)return false;if(!*a)return true;a++;b++;}
    return false;
}
static bool same_place(const ASTNode *place,const CheckedBorrowEvent *e) {
    const ASTNode *fields[32];unsigned count=0;
    while(place && place->type==AST_FIELD_ACCESS) {
        if(count==32)return false;
        fields[count++]=place;
        place=place->as.field_access.object;
    }
    size_t budget=BORROW_PATH_LIMIT+33;
    if(!place || place->type!=AST_IDENTIFIER || !same_text(place->as.identifier,e->root,&budget))return false;
    const char *p=e->projection;
    while(count){const char *field=fields[--count]->as.field_access.field_name;
        if(!field || !budget || *p!='.')return false;
        budget--;p++;
        while(*field){if(!budget || *p!=*field)return false;budget--;p++;field++;}
    }
    return !*p;
}
const CheckedBorrowEvent *checked_borrow_lookup(const Environment *env,const ASTNode *function,
    const ASTNode *call,const ASTNode *argument,unsigned ordinal) {
    const CheckedBorrowLedger *s=env?env->checked_borrow_facts:NULL;
    if(!s || s->owner!=env || !s->sealed || s->capturing || s->disabled || s->failed ||
       !function || function->type!=AST_FUNCTION || !function->as.function.checked_affine ||
       !function->as.function.affine_check_epoch || function->as.function.affine_check_epoch==UINT64_MAX ||
       function->as.function.affine_completed_epoch!=function->as.function.affine_check_epoch ||
       !call || !argument || argument->type!=AST_CALL || argument->as.call.arg_count!=1)return NULL;
    ASTNode **args=call->type==AST_CALL?call->as.call.args:
        call->type==AST_MODULE_QUALIFIED_CALL?call->as.module_qualified_call.args:NULL;
    int count=call->type==AST_CALL?call->as.call.arg_count:
        call->type==AST_MODULE_QUALIFIED_CALL?call->as.module_qualified_call.arg_count:0;
    if(!args || count<0 || ordinal>=(unsigned)count || args[ordinal]!=argument)return NULL;
    const ASTNode *expression;const char *name,*owner;
    if(!callee(call,&expression,&name,&owner))return NULL;
    for(size_t i=0;i<s->count;i++) {
        const CheckedBorrowEvent *e=&s->events[i].event;
        size_t callee_budget=BORROW_PATH_LIMIT+2;
        if(e->function==function && e->call==call && e->argument==argument && e->ordinal==ordinal &&
           e->callee_kind==call->type && e->callee_expression==expression &&
           same_text(name,e->callee_name,&callee_budget) && same_text(owner,e->callee_owner,&callee_budget) &&
           e->epoch==function->as.function.affine_check_epoch && e->place==argument->as.call.args[0] &&
           e->mode==(unsigned)argument->as.call.borrow_mode &&
           e->call_line==(uint32_t)call->line && e->call_column==(uint32_t)call->column &&
           e->argument_line==(uint32_t)argument->line && e->argument_column==(uint32_t)argument->column &&
           same_place(e->place,e))return e;
    }
    return NULL;
}

/* I retain only a real checked outer field read, never its path syntax visits. */
bool checked_field_read_capture(Environment *env,ASTNode *function,ASTNode *field,
    ASTNode *root_node,unsigned category,const char *root,const char *projection) {
    CheckedBorrowLedger *s=env?env->checked_borrow_facts:NULL;
    if(!s || !s->capturing || s->disabled)return true;
    if(s->failed)return false;
    if(!function || function->type!=AST_FUNCTION || !field || field->type!=AST_FIELD_ACCESS ||
       !root_node || root_node->type!=AST_IDENTIFIER || category<1 || category>3 ||
       field->line<=0 || field->column<=0 || !function->as.function.affine_check_epoch ||
       function->as.function.affine_check_epoch==UINT64_MAX)return true;
    if(!charge(s,1))return true;
    CheckedFieldReadEvent event={.function=function,.field=field,.root_node=root_node,
        .epoch=function->as.function.affine_check_epoch,.line=(uint32_t)field->line,
        .column=(uint32_t)field->column,.category=category};
    const ASTNode *node=field;
    while(node && node->type==AST_FIELD_ACCESS) {
        if(!charge(s,1))return true;
        if(event.depth==32){disable(s);return true;}
        event.nodes[event.depth++]=node;node=node->as.field_access.object;
    }
    if(node!=root_node)return true;
    const char *path=env->current_file;size_t np,nr,nj;
    if(!length(s,path,&np) || !length(s,root,&nr) || !length(s,projection,&nj))return true;
    if(!charge(s,np+nr+nj))return true;
    if(!np || !nr || !nj || projection[0]!='.' || !utf8(path,np) ||
       !utf8(root,nr) || !utf8(projection,nj))return true;
    if(nr>BORROW_PATH_LIMIT || nj>BORROW_PATH_LIMIT-nr ||
       np>BORROW_BYTE_LIMIT-s->total_bytes || nr>BORROW_BYTE_LIMIT-s->total_bytes-np ||
       nj>BORROW_BYTE_LIMIT-s->total_bytes-np-nr || s->total_events==BORROW_EVENT_LIMIT){disable(s);return true;}
    size_t size=np+nr+nj+3;
    if(size>BORROW_BYTE_LIMIT-s->total_bytes){disable(s);return true;}
    for(size_t i=0;i<s->read_count;i++) {
        if(!charge(s,1))return true;
        if(s->reads[i].event.function==function && s->reads[i].event.field==field)return true;
    }
    if(s->read_count==s->read_capacity) {
        size_t cap=s->read_capacity?s->read_capacity*2:16;
        if(cap>BORROW_EVENT_LIMIT)cap=BORROW_EVENT_LIMIT;
        if(cap>SIZE_MAX/sizeof(*s->reads)){s->failed=true;return false;}
        void *next=realloc(s->reads,cap*sizeof(*s->reads));
        if(!next){s->failed=true;return false;}s->reads=next;s->read_capacity=cap;
    }
    char *bytes=malloc(size);
    if(!bytes){s->failed=true;return false;}
    memcpy(bytes,path,np+1);memcpy(bytes+np+1,root,nr+1);memcpy(bytes+np+nr+2,projection,nj+1);
    event.source_path=bytes;event.root=bytes+np+1;event.projection=bytes+np+nr+2;
    s->reads[s->read_count++]=(OwnedFieldRead){event,bytes};
    s->total_events++;s->total_bytes+=size;return true;
}
const CheckedFieldReadEvent *checked_field_read_lookup(const Environment *env,
    const ASTNode *function,const ASTNode *field) {
    const CheckedBorrowLedger *s=env?env->checked_borrow_facts:NULL;
    if(!s || s->owner!=env || !s->sealed || s->capturing || s->disabled || s->failed ||
       !function || function->type!=AST_FUNCTION || !function->as.function.checked_affine ||
       !function->as.function.affine_check_epoch || function->as.function.affine_check_epoch==UINT64_MAX ||
       function->as.function.affine_completed_epoch!=function->as.function.affine_check_epoch ||
       !field || field->type!=AST_FIELD_ACCESS)return NULL;
    for(size_t i=0;i<s->read_count;i++) {
        const CheckedFieldReadEvent *e=&s->reads[i].event;
        if(e->function!=function || e->field!=field || e->epoch!=function->as.function.affine_check_epoch ||
           e->line!=(uint32_t)field->line || e->column!=(uint32_t)field->column)continue;
        const ASTNode *node=field;unsigned depth=0;
        while(node && node->type==AST_FIELD_ACCESS && depth<e->depth && node==e->nodes[depth]) {
            depth++;node=node->as.field_access.object;
        }
        if(depth!=e->depth || node!=e->root_node)continue;
        CheckedBorrowEvent place={.root=e->root,.projection=e->projection};
        if(same_place(field,&place))return e;
    }
    return NULL;
}

/* I capture only after the existing destination and RHS checks succeed. */
bool checked_field_write_capture(Environment *env,ASTNode *function,ASTNode *assignment,
    const char *root,const char *projection) {
    CheckedBorrowLedger *s=env?env->checked_borrow_facts:NULL;
    if(!s || !s->capturing || s->disabled)return true;
    if(s->failed)return false;
    if(!function || function->type!=AST_FUNCTION || !assignment || assignment->type!=AST_SET ||
       !assignment->as.set.value || assignment->line<=0 || assignment->column<=0 ||
       !function->as.function.affine_check_epoch || function->as.function.affine_check_epoch==UINT64_MAX)return true;
    if(!charge(s,1))return true;
    CheckedFieldWriteEvent event={.function=function,.assignment=assignment,.rhs=assignment->as.set.value,
        .rhs_kind=assignment->as.set.value->type,
        .epoch=function->as.function.affine_check_epoch,.line=(uint32_t)assignment->line,
        .column=(uint32_t)assignment->column};
    const char *path=env->current_file;size_t np,nr,nj;
    if(!length(s,path,&np) || !length(s,root,&nr) || !length(s,projection,&nj))return true;
    if(!charge(s,np+nr+nj))return true;
    if(!np || !nr || !nj || projection[0]!='.' || !utf8(path,np) ||
       !utf8(root,nr) || !utf8(projection,nj))return true;
    if(nr>BORROW_PATH_LIMIT || nj>BORROW_PATH_LIMIT-nr ||
       np>BORROW_BYTE_LIMIT-s->total_bytes || nr>BORROW_BYTE_LIMIT-s->total_bytes-np ||
       nj>BORROW_BYTE_LIMIT-s->total_bytes-np-nr || s->total_events==BORROW_EVENT_LIMIT){disable(s);return true;}
    size_t size=np+nr+nj+3;
    if(size>BORROW_BYTE_LIMIT-s->total_bytes){disable(s);return true;}
    for(size_t i=0;i<s->write_count;i++) {
        if(!charge(s,1))return true;
        if(s->writes[i].event.function==function && s->writes[i].event.assignment==assignment)return true;
    }
    if(s->write_count==s->write_capacity) {
        size_t cap=s->write_capacity?s->write_capacity*2:16;
        if(cap>BORROW_EVENT_LIMIT)cap=BORROW_EVENT_LIMIT;
        if(cap>SIZE_MAX/sizeof(*s->writes)){s->failed=true;return false;}
        void *next=realloc(s->writes,cap*sizeof(*s->writes));
        if(!next){s->failed=true;return false;}s->writes=next;s->write_capacity=cap;
    }
    char *bytes=malloc(size);
    if(!bytes){s->failed=true;return false;}
    memcpy(bytes,path,np+1);memcpy(bytes+np+1,root,nr+1);memcpy(bytes+np+nr+2,projection,nj+1);
    event.source_path=bytes;event.root=bytes+np+1;event.projection=bytes+np+nr+2;
    s->writes[s->write_count++]=(OwnedFieldWrite){event,bytes};
    s->total_events++;s->total_bytes+=size;return true;
}
const CheckedFieldWriteEvent *checked_field_write_lookup(const Environment *env,
    const ASTNode *function,const ASTNode *assignment) {
    const CheckedBorrowLedger *s=env?env->checked_borrow_facts:NULL;
    if(!s || s->owner!=env || !s->sealed || s->capturing || s->disabled || s->failed ||
       !function || function->type!=AST_FUNCTION || !function->as.function.checked_affine ||
       !function->as.function.affine_check_epoch || function->as.function.affine_check_epoch==UINT64_MAX ||
       function->as.function.affine_completed_epoch!=function->as.function.affine_check_epoch ||
       !assignment || assignment->type!=AST_SET || !assignment->as.set.value)return NULL;
    for(size_t i=0;i<s->write_count;i++) {
        const CheckedFieldWriteEvent *e=&s->writes[i].event;
        if(e->function!=function || e->assignment!=assignment || e->rhs!=assignment->as.set.value || e->rhs_kind!=assignment->as.set.value->type ||
           e->epoch!=function->as.function.affine_check_epoch || e->line!=(uint32_t)assignment->line ||
           e->column!=(uint32_t)assignment->column)continue;
        size_t budget=BORROW_PATH_LIMIT+2;
        if(same_text(assignment->as.set.name,e->root,&budget) &&
           e->projection[0]=='.' && same_text(assignment->as.set.field_name,e->projection+1,&budget))return e;
    }
    return NULL;
}
