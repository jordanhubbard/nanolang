#include "nsi_file_plan.h"
#include "nsi_file_catalog.h"
#include "nsi_cap.h"
#include <stdlib.h>
#include <string.h>

#define IFACE "nsi:nanolang/filesystem"
#define ID(s) IFACE "#" s
#define INT "nsi:core/int"
#define BOOL "nsi:core/bool"
#define COUNT(a) (sizeof(a)/sizeof((a)[0]))
#define MEMBER(t,n,k,d) {ID(t "." n),n,k,d}
static const NlFilePlanMember error_fields[]={
    MEMBER("FileError","status",INT,NL_FILE_DOMAIN_NONE),
    MEMBER("FileError","host_errno",INT,NL_FILE_DOMAIN_NONE),
    MEMBER("FileError","cleanup_errno",INT,NL_FILE_DOMAIN_NONE),
    MEMBER("FileError","bytes",INT,NL_FILE_DOMAIN_NONE),
    MEMBER("FileError","eof",BOOL,NL_FILE_DOMAIN_NONE),
    MEMBER("FileError","consumed",BOOL,NL_FILE_DOMAIN_NONE),
    MEMBER("FileError","cleanup_failed",BOOL,NL_FILE_DOMAIN_NONE)
};
static const NlFilePlanMember byte_fields[]={
    MEMBER("ReadByte","value",INT,NL_FILE_DOMAIN_BYTE_INT),
    MEMBER("ReadByte","eof",BOOL,NL_FILE_DOMAIN_NONE)
};
#define RESULT(t,ok) static const NlFilePlanMember t##_cases[]={ \
    MEMBER(#t,"Ok",ok,NL_FILE_DOMAIN_NONE), \
    MEMBER(#t,"Error",ID("FileError"),NL_FILE_DOMAIN_NONE) }
RESULT(OpenResult,ID("File"));
RESULT(WriteResult,INT);
RESULT(PositionResult,NULL);
RESULT(ReadResult,ID("ReadByte"));
RESULT(CloseResult,NULL);
#define TYPE(t,k,a) {ID(#t),#t,k,a,COUNT(a)}
static const NlFilePlanType types[]={
    {ID("File"),"File",NL_NSI_TYPE_RESOURCE,NULL,0},
    TYPE(FileError,NL_NSI_TYPE_RECORD,error_fields),
    TYPE(ReadByte,NL_NSI_TYPE_RECORD,byte_fields),
    TYPE(OpenResult,NL_NSI_TYPE_VARIANT,OpenResult_cases),
    TYPE(WriteResult,NL_NSI_TYPE_VARIANT,WriteResult_cases),
    TYPE(PositionResult,NL_NSI_TYPE_VARIANT,PositionResult_cases),
    TYPE(ReadResult,NL_NSI_TYPE_VARIANT,ReadResult_cases),
    TYPE(CloseResult,NL_NSI_TYPE_VARIANT,CloseResult_cases)
};
#define PARAM(m,n,t,dir,own,life,mut,dom) {ID(m "." n),n,t,dir,own,life,mut,dom}
#define BORROW(m) PARAM(m,"file",ID("File"),NL_NSI_DIR_IN,NL_NSI_OWN_BORROW,NL_NSI_LIFE_CALL,NL_NSI_MUT_MUTABLE,NL_FILE_DOMAIN_NONE)
#define RETURN(m,t) PARAM(m,"result",ID(t),NL_NSI_DIR_RETURN,NL_NSI_OWN_COPY,NL_NSI_LIFE_CALLER,NL_NSI_MUT_IMMUTABLE,NL_FILE_DOMAIN_NONE)
static const NlFilePlanParam temp_params[]={
    PARAM("temp","result",ID("OpenResult"),NL_NSI_DIR_RETURN,NL_NSI_OWN_TRANSFER,NL_NSI_LIFE_RESOURCE,NL_NSI_MUT_IMMUTABLE,NL_FILE_DOMAIN_NONE)
};
static const NlFilePlanParam write_params[]={BORROW("write_byte"),
    PARAM("write_byte","value",INT,NL_NSI_DIR_IN,NL_NSI_OWN_COPY,NL_NSI_LIFE_CALL,NL_NSI_MUT_IMMUTABLE,NL_FILE_DOMAIN_BYTE_INT),
    RETURN("write_byte","WriteResult")};
static const NlFilePlanParam rewind_params[]={BORROW("rewind"),RETURN("rewind","PositionResult")};
static const NlFilePlanParam read_params[]={BORROW("read_byte"),RETURN("read_byte","ReadResult")};
static const NlFilePlanParam close_params[]={
    PARAM("close","file",ID("File"),NL_NSI_DIR_IN,NL_NSI_OWN_TRANSFER,NL_NSI_LIFE_CALLEE,NL_NSI_MUT_IMMUTABLE,NL_FILE_DOMAIN_NONE),
    RETURN("close","CloseResult")};
#define METHOD(n,rights,acquired,mode,p,state,owned) \
    {ID(n),n,"nsi_nanolang_filesystem_" n,"nsi.local-file.v1." n,1,rights,acquired,mode,p,COUNT(p),{{state,owned},{state,NULL}}}
static const NlFilePlanMethod methods[]={
    METHOD("temp",0,NL_CAP_READ|NL_CAP_WRITE|NL_CAP_TRANSFER,NL_FILE_INPUT_NONE,temp_params,NL_FILE_OWNER_NONE,ID("File")),
    METHOD("write_byte",NL_CAP_WRITE,0,NL_FILE_INPUT_EXCLUSIVE,write_params,NL_FILE_OWNER_PRESERVED,NULL),
    METHOD("rewind",NL_CAP_READ,0,NL_FILE_INPUT_EXCLUSIVE,rewind_params,NL_FILE_OWNER_PRESERVED,NULL),
    METHOD("read_byte",NL_CAP_READ,0,NL_FILE_INPUT_EXCLUSIVE,read_params,NL_FILE_OWNER_PRESERVED,NULL),
    METHOD("close",0,0,NL_FILE_INPUT_CONSUME,close_params,NL_FILE_OWNER_CONSUMED,NULL)
};
struct NlFilePlan { const NlFilePlanMethod *methods; const NlFilePlanType *types; };
static bool text_equal(const char *a,const char *b) {
    return (!a || !b) ? a==b : strcmp(a,b)==0;
}
static bool param_equal(const NlNsiParam *a,const NlFilePlanParam *b) {
    return text_equal(a->id,b->id) && text_equal(a->name,b->name) &&
        text_equal(a->type_id,b->type_id) && a->direction==b->direction &&
        a->ownership==b->ownership && a->lifetime==b->lifetime &&
        a->mutability==b->mutability && a->optional==0 && a->streaming==NL_NSI_STREAM_NONE;
}
static bool document_equal(const NlNsi *n) {
    if(!n || n->version!=NL_NSI_VERSION || !text_equal(n->iface.id,IFACE) ||
       !text_equal(n->iface.name,"filesystem") || n->method_count!=COUNT(methods) ||
       n->type_count!=COUNT(types) || n->error_count!=1 || n->capability_count!=1 ||
       !n->methods || !n->types || !n->errors || !n->capabilities) return false;
    if(!text_equal(n->errors[0].id,ID("io")) || !text_equal(n->errors[0].name,"io") ||
       !text_equal(n->errors[0].version,"1") ||
       !text_equal(n->capabilities[0].id,"cap:nanolang/filesystem.temp") ||
       !text_equal(n->capabilities[0].name,"temp")) return false;
    for(size_t i=0;i<COUNT(methods);i++) {
        const NlNsiMethod *a=&n->methods[i];const NlFilePlanMethod *b=&methods[i];
        if(!text_equal(a->id,b->id) || !text_equal(a->name,b->name) || a->idempotent!=0 ||
           a->param_count!=b->param_count || !a->params) return false;
        for(size_t j=0;j<b->param_count;j++) if(!param_equal(&a->params[j],&b->params[j])) return false;
    }
    for(size_t i=0;i<COUNT(types);i++) {
        const NlNsiType *a=&n->types[i];const NlFilePlanType *b=&types[i];
        if(!text_equal(a->id,b->id) || !text_equal(a->name,b->name) || a->kind!=b->kind ||
           a->member_count!=b->member_count || a->element_id || a->method_id || a->result_id ||
           (b->member_count && !a->members) || (!b->member_count && a->members)) return false;
        for(size_t j=0;j<b->member_count;j++) {
            const NlNsiMember *x=&a->members[j];const NlFilePlanMember *y=&b->members[j];
            if(!text_equal(x->id,y->id) || !text_equal(x->name,y->name) ||
               !text_equal(x->type_id,y->type_id)) return false;
        }
    }
    return true;
}
NlFilePlanStatus nl_file_plan_build(const NlNsi *n,NlFilePlan **out) {
    if(!out || !document_equal(n)) return NL_FILE_PLAN_INVALID;
    NlFilePlan *p=malloc(sizeof(*p));
    if(!p) return NL_FILE_PLAN_MEMORY;
    p->methods=methods;p->types=types;*out=p;
    return NL_FILE_PLAN_OK;
}
void nl_file_plan_free(NlFilePlan *p) { free(p); }
const char *nl_file_plan_interface(const NlFilePlan *p) { return p ? IFACE : NULL; }
size_t nl_file_plan_method_count(const NlFilePlan *p) { return p ? COUNT(methods) : 0; }
size_t nl_file_plan_type_count(const NlFilePlan *p) { return p ? COUNT(types) : 0; }
const NlFilePlanMethod *nl_file_plan_method(const NlFilePlan *p,size_t i) {
    return p && i<COUNT(methods) ? &p->methods[i] : NULL;
}
const NlFilePlanType *nl_file_plan_type(const NlFilePlan *p,size_t i) {
    return p && i<COUNT(types) ? &p->types[i] : NULL;
}

const char *nl_file_catalog_interface(void) { return IFACE; }
const NlFilePlanMethod *nl_file_catalog_method(size_t i) {
    return i<COUNT(methods) ? &methods[i] : NULL;
}
