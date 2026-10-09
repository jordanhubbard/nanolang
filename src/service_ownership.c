#include "service_ownership.h"
#include <stdlib.h>
#include <string.h>

#define SO_LOCALS 1024u
#define SO_FACTS 65536u
#define SO_DEPTH 128u
typedef struct {
    const char *name;
    uint32_t id, category, mode, shared;
    bool moved, exclusive;
} SoLocal;
typedef struct { SoLocal *locals; size_t count; bool next; } SoState;
typedef struct {
    const NlServiceNamespace *space;
    const NlServiceBodyCheck *bodies;
    NlServiceOwnershipCheck *out;
    uint32_t next_binding;
    unsigned depth;
} SoCheck;
typedef struct { size_t local; unsigned mode; } SoHold;

static void so_fail(SoCheck *c, const ASTNode *node, unsigned status, const char *why) {
    if (!c->out->status) {
        c->out->status=status; c->out->diagnostic=why;
        c->out->line=node?node->line:0; c->out->column=node?node->column:0;
    }
}
static bool so_owned(uint32_t category) { return category==1 || category==2; }
static const NlServiceBodyFact *so_type(SoCheck *c, const ASTNode *node) {
    for(size_t i=0;i<c->bodies->count;++i) if(c->bodies->facts[i].node==node) return &c->bodies->facts[i];
    return NULL;
}
static int so_find(const SoState *state, const char *name) {
    for(size_t i=state->count;i>0;--i) if(!strcmp(state->locals[i-1].name,name)) return (int)i-1;
    return -1;
}
static void so_note(SoCheck *c, const ASTNode *node, unsigned action, uint32_t binding, unsigned mode) {
    if(c->out->status) return;
    if(c->out->count==SO_FACTS) { so_fail(c,node,3,"I exceeded my service ownership fact bound."); return; }
    c->out->facts[c->out->count++]=(NlServiceOwnershipFact){node,action,binding,mode};
}
static void so_bind(SoCheck *c, SoState *s, const ASTNode *node, const char *name, uint32_t category, unsigned mode) {
    if(s->count==SO_LOCALS || c->next_binding==SO_FACTS) { so_fail(c,node,3,"I exceeded my service ownership binding bound."); return; }
    SoLocal item={.name=name,.id=++c->next_binding,.category=category,.mode=mode};
    s->locals[s->count++]=item;
    if(so_owned(category)) so_note(c,node,NL_SERVICE_BIND,item.id,mode);
}
static bool so_clone(SoCheck *c, const ASTNode *node, const SoState *source, SoState *out) {
    SoLocal *locals=calloc(SO_LOCALS,sizeof *locals);
    if(!locals) { so_fail(c,node,3,"I cannot allocate service ownership branch state."); return false; }
    memcpy(locals,source->locals,source->count*sizeof *locals);
    *out=(SoState){locals,source->count,source->next}; return true;
}
static void so_exit(SoCheck *c, const SoState *s, const ASTNode *node, size_t first) {
    for(size_t i=first;i<s->count;++i) {
        const SoLocal *v=&s->locals[i];
        if(so_owned(v->category) && !v->mode && !v->moved)
            so_fail(c,node,1,"I require every File owner to be consumed before scope exit.");
    }
}
static void so_same(SoCheck *c, const ASTNode *node, const SoState *a, const SoState *b) {
    if(a->count!=b->count) { so_fail(c,node,1,"I require matching lexical ownership states."); return; }
    for(size_t i=0;i<a->count;++i) if(so_owned(a->locals[i].category) &&
        (a->locals[i].moved!=b->locals[i].moved || a->locals[i].shared!=b->locals[i].shared ||
         a->locals[i].exclusive!=b->locals[i].exclusive))
        so_fail(c,node,1,"I require branches and loop edges to agree on File ownership.");
}
static void so_expr(SoCheck *, SoState *, const ASTNode *, bool);
static void so_call(SoCheck *c, SoState *s, const ASTNode *node, ASTNode **args, int count, bool move) {
    SoHold *holds=count?calloc((size_t)count,sizeof *holds):NULL;
    if(count && !holds) { so_fail(c,node,3,"I cannot allocate service call borrow state."); return; }
    size_t held=0;
    for(int i=0;i<count && s->next && !c->out->status;++i) {
        const ASTNode *arg=args[i]; const NlServiceBodyFact *fact=so_type(c,arg);
        if(fact && fact->borrow_mode) {
            const ASTNode *root=arg;
            if(arg->type==AST_CALL && arg->as.call.borrow_mode && arg->as.call.arg_count==1) root=arg->as.call.args[0];
            int index=root->type==AST_IDENTIFIER?so_find(s,root->as.identifier):-1;
            if(index<0) { so_fail(c,arg,1,"I require a lexical File borrow root."); break; }
            SoLocal *v=&s->locals[index]; unsigned mode=fact->borrow_mode;
            if(v->moved || v->exclusive || (mode==2 && (v->shared || v->mode==1))) {
                so_fail(c,arg,1,"I cannot overlap an exclusive borrow or borrow a moved File."); break;
            }
            if(mode==2) v->exclusive=true; else ++v->shared;
            holds[held++]=(SoHold){(size_t)index,mode};
            so_note(c,arg,NL_SERVICE_BORROW,v->id,mode);
        } else so_expr(c,s,arg,true);
    }
    while(held) {
        SoHold h=holds[--held]; SoLocal *v=&s->locals[h.local];
        if(h.mode==2) v->exclusive=false; else --v->shared;
        so_note(c,node,NL_SERVICE_END_BORROW,v->id,h.mode);
    }
    free(holds);
    const NlServiceBodyFact *result=so_type(c,node);
    if(!result) so_fail(c,node,2,"I require retained service call result facts.");
    if(s->next && result && so_owned(result->type.service_category) && !move)
        so_fail(c,node,1,"I require an owner for a File service or helper result.");
}
static void so_match(SoCheck *c, SoState *s, const ASTNode *node, bool move) {
    const NlServiceBodyFact *input=so_type(c,node->as.match_expr.expr);
    if(!input) { so_fail(c,node,2,"I require retained Result ownership facts."); return; }
    so_expr(c,s,node->as.match_expr.expr,true);
    if(!s->next || c->out->status) return;
    SoState joined={0}; bool have=false;
    for(int i=0;i<node->as.match_expr.arm_count && !c->out->status;++i) {
        SoState arm={0}; if(!so_clone(c,node,s,&arm)) break;
        TypeInfo payload={0};
        if(!nl_service_member_type(c->space,&input->type,node->as.match_expr.pattern_variants[i],&payload))
            so_fail(c,node,1,"I require the exact Result ownership arm.");
        const char *name=node->as.match_expr.pattern_bindings[i];
        if(payload.base_type!=TYPE_VOID) {
            so_bind(c,&arm,node,name,payload.service_category,0);
            if(so_owned(payload.service_category) && arm.count>s->count)
                so_note(c,node,NL_SERVICE_REFINE,arm.locals[arm.count-1].id,(unsigned)i);
        }
        so_expr(c,&arm,node->as.match_expr.arm_bodies[i],move);
        if(arm.next) so_exit(c,&arm,node,s->count);
        arm.count=s->count;
        if(!have) { joined=arm; have=true; }
        else {
            if(joined.next && arm.next) so_same(c,node,&joined,&arm);
            if(!joined.next && arm.next) { free(joined.locals); joined=arm; }
            else free(arm.locals);
        }
    }
    if(have) { memcpy(s->locals,joined.locals,s->count*sizeof *s->locals); s->next=joined.next; free(joined.locals); }
}
static void so_impl(SoCheck *c, SoState *s, const ASTNode *node, bool move) {
    switch(node->type) {
    case AST_NUMBER: case AST_BOOL: return;
    case AST_IDENTIFIER: {
        int index=so_find(s,node->as.identifier);
        if(index<0) { so_fail(c,node,2,"I require a retained lexical service value."); return; }
        SoLocal *v=&s->locals[index];
        if(!so_owned(v->category)) return;
        if(v->moved || v->exclusive || (move && (v->mode || v->shared))) {
            so_fail(c,node,1,"I cannot use a moved File or consume a borrowed owner."); return;
        }
        if(move) { v->moved=true; so_note(c,node,NL_SERVICE_MOVE,v->id,0); }
        return;
    }
    case AST_CALL:
        if(node->as.call.borrow_mode) { so_fail(c,node,1,"I cannot store or escape a File borrow."); return; }
        so_call(c,s,node,node->as.call.args,node->as.call.arg_count,move); return;
    case AST_MODULE_QUALIFIED_CALL:
        so_call(c,s,node,node->as.module_qualified_call.args,node->as.module_qualified_call.arg_count,move); return;
    case AST_PREFIX_OP:
        for(int i=0;i<node->as.prefix_op.arg_count;++i) so_expr(c,s,node->as.prefix_op.args[i],false);
        return;
    case AST_FIELD_ACCESS: so_expr(c,s,node->as.field_access.object,false); return;
    case AST_LET: {
        so_expr(c,s,node->as.let.value,true);
        if(!s->next || c->out->status) return;
        const NlServiceBodyFact *fact=so_type(c,node->as.let.value);
        if(!fact) { so_fail(c,node,2,"I require retained service local type facts."); return; }
        so_bind(c,s,node,node->as.let.name,fact->type.service_category,fact->borrow_mode); return;
    }
    case AST_SET: {
        int index=so_find(s,node->as.set.name);
        if(index<0) { so_fail(c,node,2,"I require a retained service assignment root."); return; }
        SoLocal *v=&s->locals[index];
        if(v->mode || v->shared || v->exclusive || (so_owned(v->category) && !v->moved)) {
            so_fail(c,node,1,"I cannot overwrite a live or borrowed File owner."); return;
        }
        so_expr(c,s,node->as.set.value,true);
        if(!s->next || c->out->status) return;
        v->moved=false;
        if(so_owned(v->category)) so_note(c,node,NL_SERVICE_BIND,v->id,0);
        return;
    }
    case AST_RETURN:
        so_expr(c,s,node->as.return_stmt.value,true); so_exit(c,s,node,0); s->next=false; return;
    case AST_ASSERT: so_expr(c,s,node->as.assert.condition,false); return;
    case AST_BLOCK: {
        size_t first=s->count;
        for(int i=0;i<node->as.block.count && s->next && !c->out->status;++i)
            so_expr(c,s,node->as.block.statements[i],move && i+1==node->as.block.count);
        if(s->next) so_exit(c,s,node,first);
        s->count=first; return;
    }
    case AST_IF: {
        so_expr(c,s,node->as.if_stmt.condition,false);
        if(!s->next || c->out->status) return;
        SoState left={0},right={0};
        if(!so_clone(c,node,s,&left)) return;
        if(!so_clone(c,node,s,&right)) { free(left.locals); return; }
        so_expr(c,&left,node->as.if_stmt.then_branch,move);
        so_expr(c,&right,node->as.if_stmt.else_branch,move);
        if(left.next && right.next) so_same(c,node,&left,&right);
        SoState *chosen=left.next?&left:&right;
        memcpy(s->locals,chosen->locals,s->count*sizeof *s->locals); s->next=chosen->next;
        free(left.locals); free(right.locals); return;
    }
    case AST_WHILE: {
        SoState entry={0}; if(!so_clone(c,node,s,&entry)) return;
        so_expr(c,s,node->as.while_stmt.condition,false);
        if(!s->next || c->out->status) { free(entry.locals); return; }
        so_same(c,node,&entry,s);
        SoState body={0};
        if(so_clone(c,node,s,&body)) {
            so_expr(c,&body,node->as.while_stmt.body,false);
            if(body.next) so_same(c,node,&entry,&body);
            free(body.locals);
        }
        free(entry.locals); return;
    }
    case AST_MATCH: so_match(c,s,node,move); return;
    default: so_fail(c,node,2,"I have not retained ownership for this service source form."); return;
    }
}
static void so_expr(SoCheck *c, SoState *s, const ASTNode *node, bool move) {
    if(!node || !s->next || c->out->status) return;
    if(c->depth==SO_DEPTH) { so_fail(c,node,3,"I exceeded my service ownership nesting bound."); return; }
    ++c->depth; so_impl(c,s,node,move); --c->depth;
}
NlServiceOwnershipCheck *nl_service_check_ownership(const NlServiceNamespace *space, const NlServiceBodyCheck *bodies) {
    NlServiceOwnershipCheck *out=calloc(1,sizeof *out);
    if(!out) return NULL;
    SoCheck c={.space=space,.bodies=bodies,.out=out};
    if(!space || !bodies || bodies->status) {
        so_fail(&c,NULL,2,"I require complete nominal service body facts before ownership checking."); return out;
    }
    out->facts=calloc(SO_FACTS,sizeof *out->facts);
    SoState state={.locals=calloc(SO_LOCALS,sizeof *state.locals),.next=true};
    if(!out->facts || !state.locals) { free(state.locals); nl_service_ownership_free(out); return NULL; }
    for(uint32_t module=0;!out->status && nl_service_namespace_program(space,module);++module) {
        const ASTNode *program=nl_service_namespace_program(space,module);
        for(int i=0;i<program->as.program.count && !out->status;++i) {
            const ASTNode *node=program->as.program.items[i],*body=NULL;
            state.count=0; state.next=true; c.depth=0;
            if(node->type==AST_FUNCTION) {
                ++out->functions; body=node->as.function.body;
                for(int j=0;j<node->as.function.param_count;++j) {
                    const Parameter *p=&node->as.function.params[j]; const TypeInfo *info=p->type_info;
                    unsigned mode=p->type==TYPE_BORROW_MUT?2:p->type==TYPE_BORROW_SHARED?1:0;
                    if(mode && info) info=info->element_type;
                    so_bind(&c,&state,node,p->name,info?info->service_category:0,mode);
                }
            } else if(node->type==AST_SHADOW) { ++out->shadows; body=node->as.shadow.body; }
            if(body) { so_expr(&c,&state,body,false); if(state.next) so_exit(&c,&state,node,0); }
        }
    }
    free(state.locals); return out;
}
void nl_service_ownership_free(NlServiceOwnershipCheck *check) {
    if(check) { free(check->facts); free(check); }
}
