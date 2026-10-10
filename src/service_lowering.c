#include "service_lowering.h"
#include "string_literal_decode.h"
#include "nsi_file_catalog.h"
#include "nsi_socket_plan.h"
#include "nsi_websocket_plan.h"
#include "nanoisa/service_websocket_nominal.h"
#include "nanoisa/websocket_indirect_flow.h"
#include "nanoisa/websocket_codec.h"
#include "nanoisa/service_socket_nominal.h"
#include "nanoisa/socket_indirect_flow.h"
#include "nanoisa/services_indirect_flow.h"
#include "nanoisa/service_multi_nominal.h"
#include "nanoisa/isa.h"
#include "nanoisa/retained_layouts.h"
#include "nanoisa/service_file_nominal.h"
#include "nanoisa/file_indirect_flow.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define SL_FUNCTIONS 64u
#define SL_LOCALS 256u
#define SL_BYTES 65536u
#define SL_DEPTH 128u
/* My lexical names and wire slots are distinct: leaving a scope removes names
 * while retaining the exact slot declaration required by the bytecode. */
typedef struct { const char *name; uint16_t slot; } SlName;
typedef struct { uint8_t tag,mode; uint32_t layout; } SlType;
typedef struct {
    const ASTNode *node; uint32_t module, declaration;
    SlType result, locals[SL_LOCALS];
    uint16_t count, arity;
    bool copy_live[SL_LOCALS];
} SlFunction;
typedef struct {
    const NlServiceNamespace *space;
    const NlServiceBodyCheck *bodies;
    NvmModule *module;
    NlServiceLoweringResult result;
    NvmMultiNominalBindings catalogs;
    unsigned type_count;
    uint32_t struct_base[NVM_MULTI_NOMINAL_MAX_INSTANCES];
    SlFunction functions[SL_FUNCTIONS];
    SlFunction *fn;
    SlName names[SL_LOCALS];
    uint16_t name_count, next_ref, loans[SL_LOCALS], loan_count;
    uint16_t pending[SL_LOCALS], pending_count;
    uint32_t count, depth;
    bool next;
} Sl;
static void sl_fail(Sl *c,const ASTNode *n,unsigned status,const char *message) {
    if(!c->result.status)c->result=(NlServiceLoweringResult){status,n?n->line:0,n?n->column:0,message};
}
static void sl_wr32(uint8_t *p,uint32_t n) { for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(n>>(8*i)); }
static uint32_t sl_string(Sl *c,const char *s) {
    uint32_t result=nvm_add_string(c->module,s,(uint32_t)strlen(s));
    if(result==UINT32_MAX)sl_fail(c,NULL,4,"I cannot allocate my File source strings.");
    return result;
}
static const NlServiceBodyFact *sl_fact(Sl *c,const ASTNode *node) {
    for(size_t i=0;i<c->bodies->count;i++)if(c->bodies->facts[i].node==node)return &c->bodies->facts[i];
    sl_fail(c,node,1,"I require retained nominal facts for each lowered expression.");return NULL;
}
static uint32_t sl_instance(Sl *c,uint32_t declaration) {
    for(size_t i=0;i<nl_service_namespace_count(c->space);i++) {
        const NlServiceName *name=nl_service_namespace_name(c->space,i);
        if(name->target==declaration && name->kind==NL_SERVICE_TYPE && name->request<c->catalogs.count)return name->request;
    }
    sl_fail(c,NULL,1,"I require the original service instance identity.");return 0;
}
static const NlServicePlanType *sl_catalog_type(unsigned catalog,unsigned i) {
    return catalog==3?nl_websocket_catalog_type(i):catalog==2?nl_socket_catalog_type(i):nl_file_catalog_type(i);
}
static SlType sl_type(Sl *c,Type type,const TypeInfo *info) {
    uint8_t mode=0;
    if(type==TYPE_BORROW_MUT || type==TYPE_BORROW_SHARED) {
        mode=type==TYPE_BORROW_MUT?2:1;
        if(!info || !info->element_type) {sl_fail(c,NULL,1,"I require a retained borrow type.");return (SlType){0};}
        info=info->element_type;type=info->base_type;
    }
    SlType out={.mode=mode,.layout=NVM_V2_NO_INDEX};
    if(info && info->service_declaration) {
        NvmServiceInstance *instance=&c->catalogs.instances[sl_instance(c,info->service_declaration)];
        if(info->service_ordinal>=(instance->catalog==3?7u:instance->catalog==2?9u:8u))sl_fail(c,NULL,1,"I require a catalog type ordinal.");
        else out.layout=instance->layouts[info->service_ordinal];
        out.tag=(info->service_ordinal<3 || info->service_ordinal==8)?TAG_STRUCT:TAG_UNION;
    } else if(type==TYPE_INT)out.tag=TAG_INT;
    else if(type==TYPE_BOOL)out.tag=TAG_BOOL;
    else if(type==TYPE_STRING)out.tag=TAG_STRING;
    else if(type==TYPE_VOID)out.tag=TAG_VOID;
    else if(type==TYPE_FUNCTION)out.tag=TAG_FUNCTION;
    else sl_fail(c,NULL,2,"I have not lowered this ordinary service body type.");
    return out;
}
static SlType sl_node_type(Sl *c,const ASTNode *node) {
    const NlServiceBodyFact *f=sl_fact(c,node);
    return f?sl_type(c,f->type.base_type,&f->type):(SlType){0};
}
static bool sl_owner(Sl *c,SlType type) {
    if(type.mode)return false;
    for(uint32_t i=0;i<c->catalogs.count;i++)
        if(type.layout==c->catalogs.instances[i].layouts[0] || type.layout==c->catalogs.instances[i].layouts[3])return true;
    return false;
}
static void sl_bytes(Sl *c,const uint8_t *bytes,uint32_t n) {
    if(c->result.status)return;
    if(n>SL_BYTES-c->module->code_size) {sl_fail(c,NULL,3,"I exceeded my File source code bound.");return;}
    if(nvm_append_code(c->module,bytes,n)==UINT32_MAX)sl_fail(c,NULL,4,"I cannot allocate my File source code.");
}
static void sl_op(Sl *c,uint8_t op) {sl_bytes(c,&op,1);}
static void sl_u16(Sl *c,uint16_t n) {uint8_t bytes[]={(uint8_t)n,(uint8_t)(n>>8)};sl_bytes(c,bytes,2);}
static void sl_u32(Sl *c,uint32_t n) {uint8_t bytes[4];sl_wr32(bytes,n);sl_bytes(c,bytes,4);}
static void sl_local_op(Sl *c,uint8_t op,uint16_t n) {sl_op(c,op);sl_u16(c,n);}
static uint32_t sl_branch(Sl *c,uint8_t op,uint16_t local) {
    uint32_t at=c->module->code_size;sl_op(c,op);
    if(op==OP_FILE_RESULT_BRANCH)sl_u16(c,local);
    sl_u32(c,0);return at;
}
static void sl_target(Sl *c,uint32_t at,uint32_t to) {
    if(c->result.status)return;
    unsigned width=c->module->code[at]==OP_FILE_RESULT_BRANCH?3:1;
    sl_wr32(c->module->code+at+width,to-at);
}
static uint16_t sl_bind(Sl *c,const char *name,SlType type) {
    if(c->fn->count==SL_LOCALS || (name && c->name_count==SL_LOCALS)) {
        sl_fail(c,c->fn->node,3,"I exceeded my File source local bound.");return 0;
    }
    uint16_t slot=c->fn->count++;c->fn->locals[slot]=type;
    if(name)c->names[c->name_count++]=(SlName){name,slot};
    return slot;
}
static uint16_t sl_lookup(Sl *c,const ASTNode *node,const char *name) {
    for(uint16_t i=c->name_count;i;i--)if(!strcmp(c->names[i-1].name,name))return c->names[i-1].slot;
    sl_fail(c,node,1,"I require a lexical File source binding.");return 0;
}
static void sl_clear(Sl *c,uint16_t slot) {
    SlType type=c->fn->locals[slot];
    if(c->fn->copy_live[slot] && !sl_owner(c,type) && !type.mode && type.tag!=TAG_VOID) {
        sl_op(c,OP_PUSH_VOID);sl_local_op(c,OP_STORE_LOCAL,slot);
        c->fn->copy_live[slot]=false;
    }
}
static void sl_scope_end(Sl *c,uint16_t names,uint16_t slots) {
    if(c->next)for(uint16_t i=slots;i<c->fn->count;i++)sl_clear(c,i);
    c->name_count=names;
}
static void sl_store(Sl *c,uint16_t slot) {
    if(c->fn->locals[slot].tag==TAG_VOID){sl_op(c,OP_POP);return;}
    sl_local_op(c,sl_owner(c,c->fn->locals[slot])?OP_OWN_STORE_LOCAL:OP_STORE_LOCAL,slot);
    c->fn->copy_live[slot]=true;
}
static void sl_expr(Sl *,const ASTNode *,bool);
static bool sl_terminal_operand(const ASTNode *n,unsigned depth) {
    if(!n)return false;
    if(depth>=SL_DEPTH)return true;
    switch(n->type) {
    case AST_RETURN:return true;
    case AST_CALL:
        for(int i=0;i<n->as.call.arg_count;i++)if(sl_terminal_operand(n->as.call.args[i],depth+1))return true;
        return false;
    case AST_MODULE_QUALIFIED_CALL:
        for(int i=0;i<n->as.module_qualified_call.arg_count;i++)if(sl_terminal_operand(n->as.module_qualified_call.args[i],depth+1))return true;
        return false;
    case AST_PREFIX_OP:
        for(int i=0;i<n->as.prefix_op.arg_count;i++)if(sl_terminal_operand(n->as.prefix_op.args[i],depth+1))return true;
        return false;
    case AST_BLOCK:
        for(int i=0;i<n->as.block.count;i++)if(sl_terminal_operand(n->as.block.statements[i],depth+1))return true;
        return false;
    case AST_MATCH:
        if(sl_terminal_operand(n->as.match_expr.expr,depth+1))return true;
        for(int i=0;i<n->as.match_expr.arm_count;i++)if(sl_terminal_operand(n->as.match_expr.arm_bodies[i],depth+1))return true;
        return false;
    case AST_IF:return sl_terminal_operand(n->as.if_stmt.condition,depth+1) || sl_terminal_operand(n->as.if_stmt.then_branch,depth+1) || sl_terminal_operand(n->as.if_stmt.else_branch,depth+1);
    case AST_WHILE:return sl_terminal_operand(n->as.while_stmt.condition,depth+1) || sl_terminal_operand(n->as.while_stmt.body,depth+1);
    case AST_LET:return sl_terminal_operand(n->as.let.value,depth+1);
    case AST_SET:return sl_terminal_operand(n->as.set.value,depth+1);
    case AST_ASSERT:return sl_terminal_operand(n->as.assert.condition,depth+1);
    case AST_STRUCT_LITERAL:
        for(int i=0;i<n->as.struct_literal.field_count;i++)if(sl_terminal_operand(n->as.struct_literal.field_values[i],depth+1))return true;
        return false;
    case AST_FIELD_ACCESS:return sl_terminal_operand(n->as.field_access.object,depth+1);
    default:return false;
    }
}
static uint16_t sl_stage(Sl *c,SlType type) {
    uint16_t slot=sl_bind(c,NULL,type);sl_store(c,slot);
    if(c->pending_count==SL_LOCALS)sl_fail(c,NULL,3,"I exceeded my pending operand bound.");
    else c->pending[c->pending_count++]=slot;
    return slot;
}
static void sl_load(Sl *c,uint16_t slot) {
    if(c->fn->locals[slot].tag==TAG_VOID)sl_op(c,OP_PUSH_VOID);
    else sl_local_op(c,sl_owner(c,c->fn->locals[slot])?OP_OWN_MOVE_LOCAL:OP_LOAD_LOCAL,slot);
}
static void sl_pending_exit(Sl *c) {
    for(uint16_t i=c->pending_count;i;i--) {
        uint16_t slot=c->pending[i-1];
        if(sl_owner(c,c->fn->locals[slot]))sl_local_op(c,OP_FILE_DROP_LOCAL,slot);
        else sl_clear(c,slot);
    }
    c->pending_count=0;
}
static void sl_end_loans(Sl *c,uint16_t start) {
    for(uint16_t i=c->loan_count;i>start;i--) {
        sl_local_op(c,OP_FILE_END_BORROW,c->loans[i-1]);sl_op(c,OP_REGION_END);
    }
    c->loan_count=start;
}
static void sl_call(Sl *c,const ASTNode *node,ASTNode **args,int count,bool want) {
    const NlServiceBodyFact *fact=sl_fact(c,node);if(!fact)return;
    if(fact->declaration==UINT32_MAX) {
        if(count!=1){sl_fail(c,node,1,"I require one checked string length operand.");return;}
        sl_expr(c,args[0],true);if(!c->next || c->result.status)return;
        sl_op(c,OP_STR_LEN);if(!want)sl_op(c,OP_POP);return;
    }
    const NlServiceName *target=NULL;
    for(size_t i=0;i<nl_service_namespace_count(c->space);i++) {
        const NlServiceName *row=nl_service_namespace_name(c->space,i);
        if(row->id==fact->declaration && row->id==row->target){target=row;break;}
    }
    bool indirect=!fact->declaration && node->type==AST_CALL;
    if(!target && !indirect){sl_fail(c,node,1,"I require an original callable identity.");return;}
    uint16_t callable_slot=0,base_pending=c->pending_count;
    if(indirect) {
        if(node->as.call.func_expr)sl_expr(c,node->as.call.func_expr,true);
        else sl_load(c,sl_lookup(c,node,node->as.call.name));
        if(!c->next || c->result.status)return;
        callable_slot=sl_stage(c,(SlType){.tag=TAG_FUNCTION,.layout=NVM_V2_NO_INDEX});
    }
    uint16_t reference=UINT16_MAX,loans=c->loan_count,borrowed=0,references[SL_LOCALS];
    uint16_t pending=base_pending,staged[SL_LOCALS],staged_count=0;
    bool stage=sl_terminal_operand(node,0);
    if(count<0 || count>(int)SL_LOCALS){sl_fail(c,node,3,"I exceeded my call operand bound.");return;}
    for(int i=0;i<count && c->next && !c->result.status;i++) {
        const NlServiceBodyFact *arg=sl_fact(c,args[i]);if(!arg)break;
        references[i]=UINT16_MAX;
        if(arg->borrow_mode) {
            if(arg->borrow_mode!=1 && arg->borrow_mode!=2) {sl_fail(c,node,2,"I have not lowered this multi-reference call.");break;}
            const ASTNode *root=args[i];
            if(root->type==AST_CALL && root->as.call.borrow_mode)root=root->as.call.args[0];
            if(root->type!=AST_IDENTIFIER){sl_fail(c,root,2,"I require a named File reference root.");break;}
            uint16_t slot=sl_lookup(c,root,root->as.identifier);
            if(c->fn->locals[slot].mode)reference=slot;
            else {
                if(c->next_ref==SL_LOCALS || c->loan_count==SL_LOCALS) {sl_fail(c,node,3,"I exceeded my File reference bound.");break;}
                reference=c->next_ref++;sl_op(c,OP_REGION_BEGIN);
                sl_local_op(c,arg->borrow_mode==1?OP_BORROW_LOCAL_SHARED:OP_BORROW_LOCAL_EXCLUSIVE,reference);sl_u16(c,slot);
                c->loans[c->loan_count++]=reference;
            }
            references[i]=reference;borrowed++;
        } else {
            sl_expr(c,args[i],true);
            if(c->next && stage)staged[staged_count++]=sl_stage(c,sl_type(c,arg->type.base_type,&arg->type));
        }
    }
    if(!c->next || c->result.status)return;
    for(uint16_t i=0;i<staged_count;i++){sl_load(c,staged[i]);sl_clear(c,staged[i]);}
    c->pending_count=pending;
    if(indirect) {
        sl_load(c,callable_slot);sl_clear(c,callable_slot);
        sl_op(c,borrowed?OP_FILE_CALL_INDIRECT_REFS:OP_CALL_INDIRECT);
        sl_u16(c,(uint16_t)count);sl_u16(c,fact->type.base_type!=TYPE_VOID);
        if(borrowed) {
            uint8_t map[SL_LOCALS*2];
            for(int i=0;i<count;i++){map[2*i]=(uint8_t)references[i];map[2*i+1]=(uint8_t)(references[i]>>8);}
            uint32_t index=nvm_add_string(c->module,(const char *)map,(uint32_t)count*2u);
            if(index==UINT32_MAX){sl_fail(c,node,4,"I cannot retain an indirect File reference map.");return;}
            sl_u32(c,index);
        }
    } else if(target->kind==NL_SERVICE_METHOD) {
        sl_op(c,OP_FILE_SERVICE);sl_u32(c,c->catalogs.instances[target->request].imports[target->ordinal]);sl_u16(c,reference);
    } else {
        uint32_t callee=0;while(callee<c->count && c->functions[callee].declaration!=target->id)callee++;
        if(callee==c->count){sl_fail(c,node,1,"I require a retained helper body.");return;}
        if(borrowed>1) {
            uint8_t map[SL_LOCALS*2];
            for(int i=0;i<count;i++){map[2*i]=(uint8_t)references[i];map[2*i+1]=(uint8_t)(references[i]>>8);}
            uint32_t index=nvm_add_string(c->module,(const char *)map,(uint32_t)count*2u);
            if(index==UINT32_MAX){sl_fail(c,node,4,"I cannot retain a File call reference map.");return;}
            sl_op(c,OP_FILE_CALL_REFS);sl_u32(c,callee);sl_u32(c,index);
        } else {
            sl_op(c,reference==UINT16_MAX?OP_CALL:OP_CALL_REF);sl_u32(c,callee);
            if(reference!=UINT16_MAX)sl_u16(c,reference);
        }
    }
    sl_end_loans(c,loans);
    bool value=fact->type.base_type!=TYPE_VOID;
    if(value && !want)sl_op(c,OP_POP);
    if(!value && want)sl_op(c,OP_PUSH_VOID);
}
static uint8_t sl_operator(TokenType op,int arity) {
    switch(op) {
    case TOKEN_PLUS:return OP_ADD;case TOKEN_MINUS:return arity==1?OP_NEG:OP_SUB;
    case TOKEN_STAR:return OP_MUL;case TOKEN_SLASH:return OP_DIV;case TOKEN_PERCENT:return OP_MOD;
    case TOKEN_EQ:return OP_EQ;case TOKEN_NE:return OP_NE;case TOKEN_LT:return OP_LT;
    case TOKEN_LE:return OP_LE;case TOKEN_GT:return OP_GT;case TOKEN_GE:return OP_GE;
    case TOKEN_AND:return OP_AND;case TOKEN_OR:return OP_OR;case TOKEN_NOT:return OP_NOT;
    default:return OP_NOP;
    }
}
static void sl_match(Sl *c,const ASTNode *node,bool want) {
    const ASTNode *input=node->as.match_expr.expr;SlType type=sl_node_type(c,input);
    sl_expr(c,input,true);if(!c->next || c->result.status)return;
    uint16_t slot=sl_bind(c,NULL,type);sl_store(c,slot);
    c->fn->copy_live[slot]=false; /* Both arms consume this temporary. */
    uint32_t error=sl_branch(c,OP_FILE_RESULT_BRANCH,slot),end=UINT32_MAX;
    bool continuation=false;uint16_t loans=c->loan_count,pending=c->pending_count;
    for(unsigned arm=0;arm<2;arm++) {
        int index=0;while(index<node->as.match_expr.arm_count && strcmp(node->as.match_expr.pattern_variants[index],arm?"Error":"Ok"))index++;
        if(index==node->as.match_expr.arm_count){sl_fail(c,node,1,"I require both Result arms.");return;}
        if(arm)sl_target(c,error,c->module->code_size);
        c->next=true;c->loan_count=loans;c->pending_count=pending;
        uint16_t names=c->name_count,slots=c->fn->count;
        sl_local_op(c,OP_FILE_RESULT_TAKE,slot);sl_op(c,(uint8_t)arm);
        TypeInfo payload;const NlServiceBodyFact *fact=sl_fact(c,input);
        if(!fact || !nl_service_member_type(c->space,&fact->type,arm?"Error":"Ok",&payload)) {sl_fail(c,node,1,"I require a catalog Result payload.");return;}
        const char *name=node->as.match_expr.pattern_bindings[index];
        if(payload.base_type==TYPE_VOID)sl_op(c,OP_POP);
        else {uint16_t bind=sl_bind(c,name,sl_type(c,payload.base_type,&payload));sl_store(c,bind);}
        sl_expr(c,node->as.match_expr.arm_bodies[index],want);
        sl_scope_end(c,names,slots);continuation|=c->next;
        if(!arm && c->next)end=sl_branch(c,OP_JMP,0);
    }
    if(end!=UINT32_MAX)sl_target(c,end,c->module->code_size);
    c->next=continuation;c->loan_count=loans;c->pending_count=pending;
}
static void sl_expr_impl(Sl *c,const ASTNode *n,bool want) {
    if(!n){if(want)sl_op(c,OP_PUSH_VOID);return;}
    if(n->type==AST_IDENTIFIER || n->type==AST_FIELD_ACCESS) {
        const NlServiceBodyFact *value=sl_fact(c,n);
        if(value && value->type.base_type==TYPE_FUNCTION && value->declaration) {
            uint32_t target=0;while(target<c->count && c->functions[target].declaration!=value->declaration)target++;
            if(target==c->count){sl_fail(c,n,1,"I require a retained helper value.");return;}
            if(want){sl_op(c,OP_FUNCREF);sl_u32(c,target);}return;
        }
    }
    switch(n->type) {
    case AST_NUMBER: {
        sl_op(c,OP_PUSH_I64);uint64_t v=(uint64_t)n->as.number;
        for(unsigned i=0;i<8;i++)sl_op(c,(uint8_t)(v>>(8*i)));
        break;
    }
    case AST_STRING: {
        size_t length=nl_string_literal_value_bytes(n->as.string_val);
        if(length>1024u*1024u){sl_fail(c,n,3,"I exceeded my service string byte bound.");return;}
        char *bytes=nl_decode_string_literal(n->as.string_val);
        if(!bytes){sl_fail(c,n,4,"I cannot copy my service string literal.");return;}
        uint32_t index=nvm_add_string(c->module,bytes,(uint32_t)length);free(bytes);
        if(index==UINT32_MAX){sl_fail(c,n,4,"I cannot retain my service string literal.");return;}
        sl_op(c,OP_PUSH_STR);for(unsigned i=0;i<4;i++)sl_op(c,(uint8_t)(index>>(8*i)));
        break;
    }
    case AST_BOOL:sl_op(c,OP_PUSH_BOOL);sl_op(c,n->as.bool_val?1:0);break;
    case AST_IDENTIFIER: {

        uint16_t slot=sl_lookup(c,n,n->as.identifier);
        sl_load(c,slot);break;
    }
    case AST_CALL:sl_call(c,n,n->as.call.args,n->as.call.arg_count,want);return;
    case AST_MODULE_QUALIFIED_CALL:sl_call(c,n,n->as.module_qualified_call.args,n->as.module_qualified_call.arg_count,want);return;
    case AST_PREFIX_OP: {
        if(n->as.prefix_op.op==TOKEN_AND || n->as.prefix_op.op==TOKEN_OR) {
            sl_expr(c,n->as.prefix_op.args[0],true);if(!c->next)return;
            bool is_or=n->as.prefix_op.op==TOKEN_OR;
            uint32_t skip=sl_branch(c,is_or?OP_JMP_TRUE:OP_JMP_FALSE,0),end=UINT32_MAX;
            uint16_t loans=c->loan_count,pending=c->pending_count;
            sl_expr(c,n->as.prefix_op.args[1],want);
            if(c->next)end=sl_branch(c,OP_JMP,0);
            sl_target(c,skip,c->module->code_size);
            if(want){sl_op(c,OP_PUSH_BOOL);sl_op(c,is_or?1:0);}
            if(end!=UINT32_MAX)sl_target(c,end,c->module->code_size);
            c->next=true;c->loan_count=loans;c->pending_count=pending;return;
        }
        uint16_t staged[2],pending=c->pending_count;
        bool stage=sl_terminal_operand(n,0);
        if(n->as.prefix_op.arg_count<1 || n->as.prefix_op.arg_count>2){sl_fail(c,n,1,"I require a checked operator arity.");return;}
        for(int i=0;i<n->as.prefix_op.arg_count && c->next;i++) {
            sl_expr(c,n->as.prefix_op.args[i],true);
            if(c->next && stage)staged[i]=sl_stage(c,sl_node_type(c,n->as.prefix_op.args[i]));
        }
        if(!c->next)return;
        if(stage)for(int i=0;i<n->as.prefix_op.arg_count;i++){sl_load(c,staged[i]);sl_clear(c,staged[i]);}
        c->pending_count=pending;
        if((n->as.prefix_op.op==TOKEN_EQ || n->as.prefix_op.op==TOKEN_NE) &&
           sl_node_type(c,n->as.prefix_op.args[0]).tag==TAG_STRING) {
            sl_op(c,OP_STR_EQ);if(n->as.prefix_op.op==TOKEN_NE)sl_op(c,OP_NOT);
        } else sl_op(c,sl_operator(n->as.prefix_op.op,n->as.prefix_op.arg_count));break;
    }
    case AST_STRUCT_LITERAL: {
        SlType type=sl_node_type(c,n);
        const NlServiceBodyFact *fact=sl_fact(c,n);if(!fact)return;
        uint32_t instance=sl_instance(c,fact->type.service_declaration);
        unsigned catalog=c->catalogs.instances[instance].catalog;
        unsigned ordinal=catalog==3?2:8,count=catalog==3?2:7;
        if((catalog!=2 && catalog!=3) || type.layout!=c->catalogs.instances[instance].layouts[ordinal] || n->as.struct_literal.field_count!=(int)count) {
            sl_fail(c,n,2,"I require a checked catalog record constructor.");return;
        }
        uint16_t slots[7],pending=c->pending_count;
        const NlServicePlanType *record=sl_catalog_type(catalog,ordinal);
        /* I evaluate source order, then load canonical field order. Staging also
         * gives terminal field expressions the same pending-value cleanup. */
        for(unsigned i=0;i<count;i++) {
            unsigned field=0;
            while(field<count && strcmp(n->as.struct_literal.field_names[i],record->members[field].name))field++;
            if(field==count){sl_fail(c,n,1,"I require an exact catalog record field.");return;}
            sl_expr(c,n->as.struct_literal.field_values[i],true);
            if(!c->next || c->result.status)return;
            slots[field]=sl_stage(c,sl_node_type(c,n->as.struct_literal.field_values[i]));
        }
        for(unsigned i=0;i<count;i++){sl_load(c,slots[i]);sl_clear(c,slots[i]);}
        c->pending_count=pending;
        sl_op(c,OP_AGG_PACK);sl_op(c,AGG_RECORD);sl_u32(c,c->struct_base[instance]+(catalog==3?2:3));sl_u16(c,0);sl_u16(c,(uint16_t)count);break;
    }
    case AST_FIELD_ACCESS: {
        const NlServiceBodyFact *object=sl_fact(c,n->as.field_access.object);if(!object)return;
        const NlFilePlanType *type=sl_catalog_type(c->catalogs.instances[sl_instance(c,object->type.service_declaration)].catalog,object->type.service_ordinal);
        size_t index=0;while(type && index<type->member_count && strcmp(type->members[index].name,n->as.field_access.field_name))index++;
        if(!type || index==type->member_count){sl_fail(c,n,1,"I require an exact catalog field.");return;}
        sl_expr(c,n->as.field_access.object,true);if(!c->next)return;
        sl_local_op(c,OP_AGG_GET,(uint16_t)index);break;
    }
    case AST_LET:
        sl_expr(c,n->as.let.value,true);if(!c->next)return;
        sl_store(c,sl_bind(c,n->as.let.name,sl_node_type(c,n)));if(want)sl_op(c,OP_PUSH_VOID);return;
    case AST_SET: {
        uint16_t slot=sl_lookup(c,n,n->as.set.name);sl_expr(c,n->as.set.value,true);if(!c->next)return;
        sl_store(c,slot);if(want)sl_op(c,OP_PUSH_VOID);return;
    }
    case AST_ASSERT:sl_expr(c,n->as.assert.condition,true);if(!c->next)return;sl_op(c,OP_ASSERT);if(want)sl_op(c,OP_PUSH_VOID);return;
    case AST_RETURN:
        sl_expr(c,n->as.return_stmt.value,c->fn->result.tag!=TAG_VOID);if(!c->next)return;
        sl_end_loans(c,0);sl_pending_exit(c);sl_op(c,OP_RET);c->next=false;return;
    case AST_BLOCK: {
        uint16_t names=c->name_count,slots=c->fn->count;
        for(int i=0;i<n->as.block.count && c->next;i++)sl_expr(c,n->as.block.statements[i],want && i+1==n->as.block.count);
        if(!n->as.block.count && want)sl_op(c,OP_PUSH_VOID);
        sl_scope_end(c,names,slots);return;
    }
    case AST_MATCH:sl_match(c,n,want);return;
    case AST_IF: {
        sl_expr(c,n->as.if_stmt.condition,true);if(!c->next)return;
        uint32_t other=sl_branch(c,OP_JMP_FALSE,0),end=UINT32_MAX;uint16_t loans=c->loan_count,pending=c->pending_count;
        sl_expr(c,n->as.if_stmt.then_branch,false);bool yes=c->next;
        if(yes)end=sl_branch(c,OP_JMP,0);
        sl_target(c,other,c->module->code_size);c->next=true;c->loan_count=loans;c->pending_count=pending;
        sl_expr(c,n->as.if_stmt.else_branch,false);
        c->next|=yes;c->loan_count=loans;c->pending_count=pending;if(end!=UINT32_MAX)sl_target(c,end,c->module->code_size);
        if(c->next && want)sl_op(c,OP_PUSH_VOID);
        return;
    }
    case AST_WHILE: {
        uint32_t start=c->module->code_size;sl_expr(c,n->as.while_stmt.condition,true);if(!c->next)return;
        uint32_t end=sl_branch(c,OP_JMP_FALSE,0);uint16_t loans=c->loan_count,pending=c->pending_count;
        sl_expr(c,n->as.while_stmt.body,false);
        if(c->next){uint32_t back=sl_branch(c,OP_JMP,0);sl_target(c,back,start);}
        sl_target(c,end,c->module->code_size);c->next=true;c->loan_count=loans;c->pending_count=pending;
        if(want)sl_op(c,OP_PUSH_VOID);
        return;
    }
    default:sl_fail(c,n,2,"I have not lowered this checked service source form.");return;
    }
    if(!want)sl_op(c,OP_POP);
}
static void sl_expr(Sl *c,const ASTNode *n,bool want) {
    if(c->result.status || !c->next)return;
    if(++c->depth>SL_DEPTH)sl_fail(c,n,3,"I exceeded my File lowering nesting bound.");
    else sl_expr_impl(c,n,want);
    --c->depth;
}
static bool sl_catalog(Sl *c) {
    const NlFileSourcePlan *plan=nl_service_namespace_plan(c->space);
    if(!plan)return false;
    for(size_t i=0;i<nl_file_source_plan_count(plan);i++) {
        NlFileSourceRow row;if(!nl_file_source_plan_row(plan,i,&row))return false;
        if(row.request>=NVM_MULTI_NOMINAL_MAX_INSTANCES)return false;
        c->catalogs.instances[row.request].catalog=row.catalog;
        if(row.request>=c->catalogs.count)c->catalogs.count=row.request+1;
    }
    if(!c->catalogs.count)return false;
    NvmV2Layout layouts[64*9]={0};NvmV2LayoutField fields[64*9][11]={0};
    for(uint32_t r=0;r<c->catalogs.count;r++) {
        NvmServiceInstance *instance=&c->catalogs.instances[r];
        unsigned catalog=instance->catalog,types=catalog==3?7:catalog==2?9:8,base=c->type_count;
        if((catalog!=1 && catalog!=2 && catalog!=3) || (catalog==3 && c->catalogs.count!=1))return false;
        c->struct_base[r]=c->module->struct_count;
        c->module->struct_count+=catalog==2?4:3;c->module->union_count+=catalog==3?4:5;
        for(unsigned i=0;i<9;i++)instance->layouts[i]=i<types?base+i:UINT32_MAX;
        uint32_t module=sl_string(c,catalog==3?nl_websocket_catalog_interface():catalog==2?nl_socket_catalog_interface():nl_file_catalog_interface());
        for(unsigned i=0;i<(catalog==3?4u:5u);i++) {
            uint8_t tags[3]={0};
            const NlServicePlanMethod *method=catalog==3?nl_websocket_catalog_method(i):catalog==2?nl_socket_catalog_method(i):nl_file_catalog_method(i);
            if(method->param_count<1 || method->param_count>4)return false;
            for(size_t j=0;j+1<method->param_count;j++) {
                const char *id=method->params[j].type_id;
                tags[j]=!strcmp(id,"nsi:core/string")?TAG_STRING:!strcmp(id,"nsi:core/int")?TAG_INT:!strcmp(id,"nsi:core/bool")?TAG_BOOL:TAG_STRUCT;
            }
            uint32_t index=nvm_add_import(c->module,module,sl_string(c,method->id),(uint16_t)(method->param_count-1),TAG_UNION,tags);
            if(index==UINT32_MAX){sl_fail(c,NULL,4,"I cannot allocate service imports.");return false;}
            c->module->imports[index].kind=NVM_IMPORT_SERVICE;instance->imports[i]=index;
        }
        for(unsigned i=0;i<types;i++) {
            const NlServicePlanType *type=sl_catalog_type(catalog,i);
            layouts[base+i]=(NvmV2Layout){.kind=(i<3 || i==8)?NVM_V2_LAYOUT_STRUCT:NVM_V2_LAYOUT_UNION,
                .name_idx=sl_string(c,type->id),.field_count=(uint16_t)type->member_count,.fields=fields[base+i]};
            for(size_t j=0;j<type->member_count;j++) {
                const char *id=type->members[j].type_id;uint8_t tag=TAG_VOID;uint32_t layout=NVM_V2_NO_INDEX;
                if(id) {
                    if(!strcmp(id,"nsi:core/int"))tag=TAG_INT;
                    else if(!strcmp(id,"nsi:core/bool"))tag=TAG_BOOL;
                    else if(!strcmp(id,"nsi:core/string"))tag=TAG_STRING;
                    else for(unsigned k=0;k<types;k++)if(!strcmp(id,sl_catalog_type(catalog,k)->id)){layout=base+k;tag=(k<3 || k==8)?TAG_STRUCT:TAG_UNION;}
                }
                fields[base+i][j]=(NvmV2LayoutField){tag,layout,sl_string(c,type->members[j].id)};
            }
        }
        c->type_count+=types;
    }
    NvmV2Layouts table={layouts,c->type_count};
    if(nvm_retain_layouts(c->module,&table)!=NVM_V2_OK){sl_fail(c,NULL,4,"I cannot retain service layouts.");return false;}
    unsigned catalog=c->catalogs.instances[0].catalog;
    size_t capacity=c->catalogs.count>1?16+64*c->catalogs.count:catalog==3?NVM_WEBSOCKET_NOMINAL_BYTES:catalog==2?NVM_SOCKET_NOMINAL_BYTES:NVM_FILE_NOMINAL_BYTES,size=0;
    c->module->service_data=malloc(capacity);
    if(!c->module->service_data){sl_fail(c,NULL,4,"I cannot retain service bindings.");return false;}
    NvmServiceResult status;
    if(c->catalogs.count>1)status=nvm_multi_nominal_encode(&c->catalogs,c->module->service_data,capacity,&size);
    else if(catalog==3) {
        NvmWebSocketNominalBindings websocket={0};
        memcpy(websocket.imports,c->catalogs.instances[0].imports,sizeof websocket.imports);memcpy(websocket.layouts,c->catalogs.instances[0].layouts,sizeof websocket.layouts);
        status=nvm_websocket_nominal_encode(&websocket,c->module->service_data,capacity,&size);
    } else if(catalog==2) {
        NvmSocketNominalBindings socket={0};
        memcpy(socket.imports,c->catalogs.instances[0].imports,sizeof socket.imports);memcpy(socket.layouts,c->catalogs.instances[0].layouts,sizeof socket.layouts);
        status=nvm_socket_nominal_encode(&socket,c->module->service_data,capacity,&size);
    } else {
        NvmFileNominalBindings file={0};
        memcpy(file.imports,c->catalogs.instances[0].imports,sizeof file.imports);memcpy(file.layouts,c->catalogs.instances[0].layouts,sizeof file.layouts);
        status=nvm_file_nominal_encode(&file,c->module->service_data,capacity,&size);
    }
    if(status!=NVM_SERVICE_OK)return false;
    c->module->service_size=(uint32_t)size;return !c->result.status;
}
static void sl_desc(uint8_t *out,SlType type) {out[0]=type.tag;out[1]=type.mode;sl_wr32(out+4,type.layout);}
static void sl_ownership(Sl *c) {
    size_t flags=(c->type_count+3u)&~3u,header=12+flags,size=header;
    for(uint32_t i=0;i<c->count;i++)size+=12+8*c->functions[i].count;
    uint8_t *bytes=calloc(1,size);if(!bytes){sl_fail(c,NULL,4,"I cannot retain File local declarations.");return;}
    sl_wr32(bytes,1);sl_wr32(bytes+4,c->type_count);
    for(unsigned i=0;i<c->type_count;i++)bytes[8+i]=sl_owner(c,(SlType){.layout=i})?3:1;
    sl_wr32(bytes+8+flags,c->count);size_t at=header;
    for(uint32_t i=0;i<c->count;i++) {
        SlFunction *f=&c->functions[i];bytes[at]=(uint8_t)f->count;bytes[at+1]=(uint8_t)(f->count>>8);
        bytes[at+2]=(uint8_t)f->arity;bytes[at+3]=(uint8_t)(f->arity>>8);sl_desc(bytes+at+4,f->result);at+=12;
        for(uint16_t j=0;j<f->count;j++){sl_desc(bytes+at,f->locals[j]);at+=8;}
    }
    c->module->ownership_data=bytes;c->module->ownership_size=(uint32_t)size;
}
NlServiceLoweringResult nl_service_lower(const NlServiceNamespace *space,
    const NlServiceBodyCheck *bodies,const NlServiceOwnershipCheck *owners,
    const ASTNode *selection,NvmModule **out) {
    if(!space || !bodies || bodies->status || !owners || owners->status || !out)
        return (NlServiceLoweringResult){1,0,0,"I require complete nominal and ownership checks before lowering."};
    Sl *c=calloc(1,sizeof *c);if(!c)return (NlServiceLoweringResult){4,0,0,"I cannot allocate File lowering state."};
    c->space=space;c->bodies=bodies;c->module=nvm_module_new();
    if(!c->module)sl_fail(c,NULL,4,"I cannot allocate a File module.");
    uint32_t selected=UINT32_MAX,root_module=0;
    while(nl_service_namespace_program(space,root_module+1))root_module++;
    if(c->module && !sl_catalog(c) && !c->result.status)sl_fail(c,NULL,1,"I require a complete File catalog plan.");
    /* I order named functions before shadows, matching my independent Nano parser. */
    for(unsigned pass=0;pass<2 && !c->result.status;pass++)
    for(uint32_t module=0;!c->result.status && nl_service_namespace_program(space,module);module++) {
        const ASTNode *program=nl_service_namespace_program(space,module);
        for(int i=0;i<program->as.program.count && !c->result.status;i++) {
            const ASTNode *node=program->as.program.items[i];
            if(node->type!=(pass==0?AST_FUNCTION:AST_SHADOW))continue;
            if(c->count==SL_FUNCTIONS-1){sl_fail(c,node,3,"I exceeded my File function bound.");break;}
            SlFunction *fn=&c->functions[c->count];fn->node=node;fn->module=module;
            const char *name=node->type==AST_FUNCTION?node->as.function.name:node->as.shadow.function_name;
            if(node->type==AST_FUNCTION) {
                const NlServiceName *row=nl_service_namespace_lookup(space,nl_service_namespace_module(space,module),name);
                if(!row){sl_fail(c,node,1,"I require the original function identity.");break;}
                fn->declaration=row->target;
                fn->result=sl_type(c,node->as.function.return_type,node->as.function.return_type_info);
                if(!selection && module==root_module && !strcmp(name,"main"))selected=c->count;
            } else fn->result=(SlType){.tag=TAG_VOID,.layout=NVM_V2_NO_INDEX};
            if(selection==node)selected=c->count;
            char label[4608];int n=snprintf(label,sizeof label,"%s::%s%s#%u",nl_service_namespace_module(space,module),node->type==AST_SHADOW?"shadow.":"",name,c->count);
            if(n<0 || (size_t)n>=sizeof label){sl_fail(c,node,3,"I exceeded my function name bound.");break;}
            NvmFunctionEntry entry={.name_idx=sl_string(c,label),.result_tag=fn->result.tag,.result_count=fn->result.tag!=TAG_VOID};
            if(nvm_add_function(c->module,&entry)!=c->count)sl_fail(c,node,4,"I cannot retain a File function.");
            c->count++;
        }
    }
    if(selected==UINT32_MAX)sl_fail(c,selection,1,"I require a selected File entry in this graph.");
    for(uint32_t i=0;i<c->count && !c->result.status;i++) {
        c->fn=&c->functions[i];const ASTNode *node=c->fn->node;c->name_count=0;c->loan_count=0;c->pending_count=0;c->next=true;
        NvmFunctionEntry *entry=&c->module->functions[i];entry->code_offset=c->module->code_size;
        if(node->type==AST_FUNCTION) {
            int count=node->as.function.param_count;
            if(count<0 || count>(int)SL_LOCALS){sl_fail(c,node,3,"I exceeded my File parameter bound.");break;}
            uint8_t tags[SL_LOCALS];
            for(int j=0;j<count;j++) {
                const Parameter *p=&node->as.function.params[j];SlType type=sl_type(c,p->type,p->type_info);
                sl_bind(c,p->name,type);tags[j]=type.tag;
            }
            c->fn->arity=entry->arity=(uint16_t)count;
            if(!nvm_set_function_param_types(c->module,i,tags,(uint16_t)count))sl_fail(c,node,4,"I cannot retain File parameters.");
        }
        c->next_ref=c->fn->arity;
        sl_expr(c,node->type==AST_FUNCTION?node->as.function.body:node->as.shadow.body,false);
        if(c->next) {
            if(c->fn->result.tag!=TAG_VOID)sl_fail(c,node,1,"I require a return from this File helper.");
            else sl_op(c,OP_RET);
        }
        entry->local_count=c->fn->count;entry->code_length=c->module->code_size-entry->code_offset;
    }
    if(!c->result.status) {
        SlFunction *entry=&c->functions[selected];
        if(entry->arity || (entry->result.tag!=TAG_INT && entry->result.tag!=TAG_BOOL && entry->node->type!=AST_SHADOW))
            sl_fail(c,entry->node,2,"I require a zero-argument scalar entry or a selected shadow.");
        else if(entry->node->type==AST_SHADOW) {
            SlFunction *wrapper=&c->functions[c->count];wrapper->result=(SlType){.tag=TAG_INT,.layout=NVM_V2_NO_INDEX};
            NvmFunctionEntry fn={.name_idx=sl_string(c,"__file_selected_shadow"),.code_offset=c->module->code_size,.result_tag=TAG_INT,.result_count=1};
            sl_op(c,OP_CALL);sl_u32(c,selected);sl_op(c,OP_PUSH_I64);for(unsigned j=0;j<8;j++)sl_op(c,0);sl_op(c,OP_RET);
            fn.code_length=c->module->code_size-fn.code_offset;
            if(nvm_add_function(c->module,&fn)!=c->count)sl_fail(c,NULL,4,"I cannot retain my shadow entry.");
            selected=c->count++;
        }
        c->module->header.entry_point=selected;c->module->header.flags|=NVM_FLAG_HAS_MAIN|NVM_FLAG_NEEDS_EXTERN;
        sl_ownership(c);
    }
    NlServiceLoweringResult result=c->result;
    if(result.status)nvm_module_free(c->module);else *out=c->module;
    free(c);return result;
}
/* I canonicalize only my checked File source product: reference maps, catalog
 * names, function names; import signatures precede helper signatures. */
static uint32_t sl_order(uint32_t *order,uint32_t count,uint32_t *next,uint32_t old) {
    if(old>=count)return old;
    if(order[old]==UINT32_MAX)order[old]=(*next)++;
    return order[old];
}
static NvmV2Result sl_canonical_wire(NvmV2Module *wire,uint8_t **owned_code) {
    uint32_t nc=wire->constants.count,ns=wire->signatures.count,cn=0,sn=0;
    uint32_t *constants=malloc((size_t)nc*sizeof *constants),*signatures=malloc((size_t)ns*sizeof *signatures);
    NvmV2Constant *pool=calloc(nc,sizeof *pool);
    NvmV2Signature *types=calloc(ns,sizeof *types);
    uint8_t *code=malloc((size_t)wire->code_size);
    NvmV2Result result=NVM_V2_ERR_TRUNCATED;
    if(!constants || !signatures || !pool || !types || !code)goto done;
    result=NVM_V2_ERR_INDEX_RANGE;
    for(uint32_t i=0;i<nc;i++)constants[i]=UINT32_MAX;
    for(uint32_t i=0;i<ns;i++)signatures[i]=UINT32_MAX;
    memcpy(code,wire->code,(size_t)wire->code_size);
    for(uint64_t pc=0;pc<wire->code_size;) {
        DecodedInstruction d;
        if(!isa_decode(code+pc,(uint32_t)(wire->code_size-pc),&d))goto done;
        if(d.opcode==OP_PUSH_STR) {
            uint32_t old=d.operands[0].u32;
            if(old>=nc)goto done;
            sl_wr32(code+pc+1,sl_order(constants,nc,&cn,old));
        }
        if(d.opcode==OP_FILE_CALL_REFS || d.opcode==OP_FILE_CALL_INDIRECT_REFS) {
            uint32_t old=d.operands[d.opcode==OP_FILE_CALL_REFS?1:2].u32;
            if(old>=nc)goto done;
            sl_wr32(code+pc+5,sl_order(constants,nc,&cn,old));
        }
        pc+=d.byte_length;
    }
    for(uint32_t i=0;i<wire->layouts.count;i++) {
        NvmV2Layout *layout=&wire->layouts.items[i];
        layout->name_idx=sl_order(constants,nc,&cn,layout->name_idx);
        for(uint16_t j=0;j<layout->field_count;j++)
            layout->fields[j].name_idx=sl_order(constants,nc,&cn,layout->fields[j].name_idx);
    }
    for(uint32_t i=0;i<wire->imports.count;i++) {
        NvmV2Import *import=&wire->imports.items[i];
        import->module_name_idx=sl_order(constants,nc,&cn,import->module_name_idx);
        import->symbol_name_idx=sl_order(constants,nc,&cn,import->symbol_name_idx);
        import->signature_idx=sl_order(signatures,ns,&sn,import->signature_idx);
    }
    for(uint32_t i=0;i<wire->functions.count;i++) {
        NvmV2Function *fn=&wire->functions.items[i];
        fn->name_idx=sl_order(constants,nc,&cn,fn->name_idx);
        fn->signature_idx=sl_order(signatures,ns,&sn,fn->signature_idx);
    }
    for(uint32_t i=0;i<nc;i++)pool[sl_order(constants,nc,&cn,i)]=wire->constants.items[i];
    for(uint32_t i=0;i<ns;i++)types[sl_order(signatures,ns,&sn,i)]=wire->signatures.items[i];
    free(wire->constants.items);wire->constants.items=pool;pool=NULL;
    free(wire->signatures.items);wire->signatures.items=types;types=NULL;
    wire->code=code;*owned_code=code;code=NULL;result=NVM_V2_OK;
done:
    free(constants);free(signatures);free(pool);free(types);free(code);return result;
}
static NvmV2Result sl_file_stack_bounds(const NvmModule *module,NvmV2Module *wire,unsigned *failure) {
    NvmFileIndirectFlow *report=NULL;
    NvmFileFlowStatus status=nvm_file_indirect_flow_analyze(module,&report);
    if(status!=NVM_FILE_FLOW_OK) {
        *failure=status==NVM_FILE_FLOW_MEMORY?4:status==NVM_FILE_FLOW_LIMIT?3:2;
        return NVM_V2_ERR_INDEX_RANGE;
    }
    NvmV2Result result=NVM_V2_OK;
    if(result==NVM_V2_OK)for(uint32_t f=0;f<module->function_count;f++) {
        NvmFileCodeFunction function;
        if(!nvm_file_indirect_flow_function(report,f,&function)){result=NVM_V2_ERR_INDEX_RANGE;break;}
        uint16_t peak=0;
        for(uint16_t i=0;i<function.instruction_count;i++) {
            uint8_t count=0;
            if(!nvm_file_indirect_flow_variant_count(report,f,i,&count)){result=NVM_V2_ERR_INDEX_RANGE;break;}
            for(uint8_t j=0;j<count;j++) {
                NvmFileCyclicVariant fact;
                if(!nvm_file_indirect_flow_variant(report,f,i,j,&fact)){result=NVM_V2_ERR_INDEX_RANGE;break;}
                if(fact.input.stack>peak)peak=fact.input.stack;
                if(fact.output.stack>peak)peak=fact.output.stack;
            }
        }
        wire->functions.items[f].max_stack=peak;
    }
    nvm_file_indirect_flow_free(report);return result;
}
static NvmV2Result sl_socket_stack_bounds(const NvmModule *module,NvmV2Module *wire,unsigned *failure) {
    NvmSocketIndirectFlow *report=NULL;
    NvmSocketFlowStatus status=nvm_socket_indirect_flow_analyze(module,&report);
    if(status!=NVM_SOCKET_FLOW_OK) {
        *failure=status==NVM_SOCKET_FLOW_MEMORY?4:status==NVM_SOCKET_FLOW_LIMIT?3:2;
        return NVM_V2_ERR_INDEX_RANGE;
    }
    NvmV2Result result=NVM_V2_OK;
    if(result==NVM_V2_OK)for(uint32_t f=0;f<module->function_count;f++) {
        NvmSocketCodeFunction function;
        if(!nvm_socket_indirect_flow_function(report,f,&function)){result=NVM_V2_ERR_INDEX_RANGE;break;}
        uint16_t peak=0;
        for(uint16_t i=0;i<function.instruction_count;i++) {
            uint8_t count=0;
            if(!nvm_socket_indirect_flow_variant_count(report,f,i,&count)){result=NVM_V2_ERR_INDEX_RANGE;break;}
            for(uint8_t j=0;j<count;j++) {
                NvmSocketCyclicVariant fact;
                if(!nvm_socket_indirect_flow_variant(report,f,i,j,&fact)){result=NVM_V2_ERR_INDEX_RANGE;break;}
                if(fact.input.stack>peak)peak=fact.input.stack;
                if(fact.output.stack>peak)peak=fact.output.stack;
            }
        }
        wire->functions.items[f].max_stack=peak;
    }
    nvm_socket_indirect_flow_free(report);return result;
}
static NvmV2Result sl_websocket_stack_bounds(const NvmModule *module,NvmV2Module *wire,unsigned *failure) {
    NvmWebSocketIndirectFlow *report=NULL;
    NvmWebSocketFlowStatus status=nvm_websocket_indirect_flow_analyze(module,&report);
    if(status!=NVM_WEBSOCKET_FLOW_OK) {
        *failure=status==NVM_WEBSOCKET_FLOW_MEMORY?4:status==NVM_WEBSOCKET_FLOW_LIMIT?3:2;
        return NVM_V2_ERR_INDEX_RANGE;
    }
    NvmV2Result result=NVM_V2_OK;
    if(result==NVM_V2_OK)for(uint32_t f=0;f<module->function_count;f++) {
        NvmWebSocketCodeFunction function;
        if(!nvm_websocket_indirect_flow_function(report,f,&function)){result=NVM_V2_ERR_INDEX_RANGE;break;}
        uint16_t peak=0;
        for(uint16_t i=0;i<function.instruction_count;i++) {
            uint8_t count=0;
            if(!nvm_websocket_indirect_flow_variant_count(report,f,i,&count)){result=NVM_V2_ERR_INDEX_RANGE;break;}
            for(uint8_t j=0;j<count;j++) {
                NvmWebSocketCyclicVariant fact;
                if(!nvm_websocket_indirect_flow_variant(report,f,i,j,&fact)){result=NVM_V2_ERR_INDEX_RANGE;break;}
                if(fact.input.stack>peak)peak=fact.input.stack;
                if(fact.output.stack>peak)peak=fact.output.stack;
            }
        }
        wire->functions.items[f].max_stack=peak;
    }
    nvm_websocket_indirect_flow_free(report);return result;
}
static NvmV2Result sl_services_stack_bounds(const NvmModule *module,NvmV2Module *wire,unsigned *failure) {
    NvmServicesIndirectFlow *report=NULL;
    NvmServicesFlowStatus status=nvm_services_indirect_flow_analyze(module,&report);
    if(status!=NVM_SERVICES_FLOW_OK) {
        *failure=status==NVM_SERVICES_FLOW_MEMORY?4:status==NVM_SERVICES_FLOW_LIMIT?3:2;
        return NVM_V2_ERR_INDEX_RANGE;
    }
    NvmV2Result result=NVM_V2_OK;
    if(result==NVM_V2_OK)for(uint32_t f=0;f<module->function_count;f++) {
        NvmServicesCodeFunction function;
        if(!nvm_services_indirect_flow_function(report,f,&function)){result=NVM_V2_ERR_INDEX_RANGE;break;}
        uint16_t peak=0;
        for(uint16_t i=0;i<function.instruction_count;i++) {
            uint8_t count=0;
            if(!nvm_services_indirect_flow_variant_count(report,f,i,&count)){result=NVM_V2_ERR_INDEX_RANGE;break;}
            for(uint8_t j=0;j<count;j++) {
                NvmServicesCyclicVariant fact;
                if(!nvm_services_indirect_flow_variant(report,f,i,j,&fact)){result=NVM_V2_ERR_INDEX_RANGE;break;}
                if(fact.input.stack>peak)peak=fact.input.stack;
                if(fact.output.stack>peak)peak=fact.output.stack;
            }
        }
        wire->functions.items[f].max_stack=peak;
    }
    nvm_services_indirect_flow_free(report);return result;
}
NlServiceLoweringResult nl_service_serialize(const NvmModule *module,uint8_t **out,size_t *size) {
    if(!module || !out || !size)return (NlServiceLoweringResult){1,0,0,"I require module and output storage."};
    NvmWebSocketNominalBindings bindings;
    bool websocket=module->service_data && nvm_websocket_nominal_decode(module->service_data,module->service_size,&bindings)==NVM_SERVICE_OK;
    NvmV2Module wire={0};NvmV2Result result=websocket?nvm_websocket_from_module(module,&wire):nvm_v2_from_nvm_module(module,&wire);
    unsigned failure=2;
    if(result==NVM_V2_OK)result=websocket?sl_websocket_stack_bounds(module,&wire,&failure):module->service_size>NVM_SOCKET_NOMINAL_BYTES?sl_services_stack_bounds(module,&wire,&failure):module->service_size==NVM_SOCKET_NOMINAL_BYTES?
        sl_socket_stack_bounds(module,&wire,&failure):sl_file_stack_bounds(module,&wire,&failure);
    size_t count=0;uint8_t *bytes=NULL;bool memory_failure=false;
    uint8_t *canonical_code=NULL;
    if(result==NVM_V2_OK) {
        result=sl_canonical_wire(&wire,&canonical_code);
        memory_failure=result==NVM_V2_ERR_TRUNCATED;
    }
    if(result==NVM_V2_OK)result=websocket?nvm_websocket_serialize(&wire,NULL,0,&count):nvm_v2_module_serialize(&wire,NULL,0,&count);
    if(result==NVM_V2_OK) {
        bytes=malloc(count);
        if(!bytes){memory_failure=true;result=NVM_V2_ERR_TRUNCATED;}
        else result=websocket?nvm_websocket_serialize(&wire,bytes,count,&count):nvm_v2_module_serialize(&wire,bytes,count,&count);
    }
    nvm_v2_module_free(&wire);free(canonical_code);
    if(result!=NVM_V2_OK){free(bytes);return (NlServiceLoweringResult){memory_failure?4u:failure,0,0,"I cannot serialize my checked File module."};}
    *out=bytes;*size=count;return (NlServiceLoweringResult){0};
}
