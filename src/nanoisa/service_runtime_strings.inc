/* I own immutable counted strings for one invocation. I intern equal contents
 * so literal loops do not allocate repeatedly. The ordinary runtime byte limit
 * includes every retained node; host results use this same checked copy path. */
static const ServiceRuntimeString *fr_string_identity(const SERVICE_TYPE(Runtime) *c,int64_t identity) {
    return identity>0 && (uint64_t)identity<=c->string_count?c->strings[identity-1]:NULL;
}
static const ServiceRuntimeString *fr_string(const SERVICE_TYPE(Runtime) *c,uint32_t root) {
    if(!fr_live(c,root))return NULL;
    const SERVICE_TYPE(RuntimeView) *v=&c->values[root].view;
    if(v->owning || v->formal || v->fields!=1 || !fr_same(v->type,fr_scalar_type(TAG_STRING)))return NULL;
    return fr_string_identity(c,v->values[0]);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_string)(SERVICE_TYPE(Runtime) *c,uint32_t dst,
    const void *bytes,size_t length) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(!fr_empty(c,dst) || (!bytes && length))return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    if(length>1024u*1024u)return fr_error(c,SERVICE_CONST(RUNTIME_LIMIT),CORE_STATUS(LIMIT));
    ServiceRuntimeString *found=NULL;
    for(uint32_t i=0;i<c->string_count;i++) {
        ServiceRuntimeString *s=c->strings[i];
        if(s->length==length && (!length || !memcmp(s->bytes,bytes,length))){found=s;break;}
    }
    if(!found) {
        size_t bound=c->storage.allocation_bound;
        if(c->string_count>=4096 || !fr_add(&bound,1,sizeof(ServiceRuntimeString)) || !fr_add(&bound,length,1))
            return fr_error(c,SERVICE_CONST(RUNTIME_LIMIT),CORE_STATUS(LIMIT));
        found=calloc(1,sizeof(*found)+length);
        if(!found)return fr_error(c,SERVICE_CONST(RUNTIME_MEMORY),CORE_STATUS(MEMORY));
        found->identity=++c->string_count;found->length=length;
        if(length)memcpy(found->bytes,bytes,length);
        c->strings[found->identity-1]=found;c->storage.allocation_bound=bound;
    }
    SERVICE_TYPE(RuntimeView) v=fr_view(fr_scalar_type(TAG_STRING));v.fields=1;v.values[0]=found->identity;
    c->values[dst].view=v;return SERVICE_CONST(RUNTIME_OK);
}
bool SERVICE_FN(runtime_string_read)(const SERVICE_TYPE(Runtime) *c,uint32_t root,
    void *out,size_t capacity,size_t *length) {
    if(!c || !length || c->busy || c->phase!=FR_ACTIVE || c->report.status!=SERVICE_CONST(RUNTIME_OK))return false;
    const ServiceRuntimeString *s=fr_string(c,root);
    if(!s || capacity<s->length || (!out && s->length))return false;
    if(s->length)memcpy(out,s->bytes,s->length);
    *length=s->length;return true;
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_string_literal)(SERVICE_TYPE(Runtime) *c,uint32_t dst) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    if(c->kind!=FR_INDIRECT || !c->frame_count)return fr_error(c,SERVICE_CONST(RUNTIME_STATE),CORE_STATUS(STATE));
    FileRuntimeFrame *f=&c->frames[c->frame_count-1];
    SERVICE_TYPE(CodeInstruction) in;SERVICE_TYPE(BodyInstruction) fact;
    const uint8_t *bytes;size_t length;
    if(!fr_instruction(c,f->function,f->instruction,f->variant,&in,&fact) || in.decoded.opcode!=OP_PUSH_STR ||
       !SERVICE_FN(indirect_hosted_string)(c->indirect_plan,in.decoded.operands[0].u32,&bytes,&length))
        return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    return SERVICE_FN(runtime_string)(c,dst,bytes,length);
}
SERVICE_TYPE(RuntimeStatus) SERVICE_FN(runtime_string_operation)(SERVICE_TYPE(Runtime) *c,uint8_t op,
    uint32_t left,uint32_t right,uint32_t dst) {
    SERVICE_TYPE(RuntimeStatus) status=fr_effect(c);if(status!=SERVICE_CONST(RUNTIME_OK))return status;
    const ServiceRuntimeString *a=fr_string(c,left),*b=op==OP_STR_EQ?fr_string(c,right):NULL;
    if(!a || (op!=OP_STR_EQ && op!=OP_STR_LEN) ||
       (op==OP_STR_EQ && (!b || left==right)) || (dst!=left && dst!=right && !fr_empty(c,dst)) ||
       (op==OP_STR_LEN && (right!=SERVICE_CONST(RUNTIME_NO_SLOT) || (dst!=left && !fr_empty(c,dst)))))
        return fr_error(c,SERVICE_CONST(RUNTIME_TYPE),CORE_STATUS(TYPE));
    int64_t value=op==OP_STR_LEN?(int64_t)a->length:
        a->length==b->length && (!a->length || !memcmp(a->bytes,b->bytes,a->length));
    fr_clear(c,left);if(op==OP_STR_EQ)fr_clear(c,right);
    return SERVICE_FN(runtime_scalar)(c,dst,op==OP_STR_LEN?TAG_INT:TAG_BOOL,value);
}
