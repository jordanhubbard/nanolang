#include "sdk_provider_validation.h"
#include "ownership_layouts_private.h"
#include <stdlib.h>
#include <string.h>
struct NvmSdkDescription {
    NvmSdkModuleSnapshot *module;
    NvmOwnershipDeclarationPlan *declarations;
    NvmSdkProviderTransport *provider;
};
typedef struct {
    NvmSdkDescription *owned;
    const NvmV2Module *m;
    NvmDeclarationCounts declarations;
    uint32_t counts[5],lifetimes[2];
    NvmPreparationBudget *budget;
    NvmSdkResult error;
    uint32_t nominal[256];
    void *pairs;
    bool identity_ready;
} SdkCheck;
static bool sdk_step(SdkCheck *c,uint32_t steps) {
    if(!nvm_preparation_charge(c->budget,0,steps)){c->error=NVM_SDK_LIMIT;return false;}
    return true;
}
static bool sdk_name(SdkCheck *c,uint32_t index,bool empty) {
    if(index>=c->m->constants.count)return false;
    const NvmV2Constant *v=&c->m->constants.items[index];
    if(v->tag!=TAG_STRING || (!empty&&!v->length) || (v->length&&!v->payload))return false;
    if(v->length>UINT32_MAX || !sdk_step(c,(uint32_t)v->length))return false;
    return !v->length || !memchr(v->payload,0,v->length);
}
static bool sdk_same_name(SdkCheck *c,uint32_t a,uint32_t b) {
    const NvmV2Constant *x=&c->m->constants.items[a],*y=&c->m->constants.items[b];
    if(x->length!=y->length)return false;
    return sdk_step(c,(uint32_t)x->length) && (!x->length || !memcmp(x->payload,y->payload,x->length));
}
static bool sdk_type(SdkCheck *c,uint32_t i,NvmOrdinaryArrayType *out) {
    return sdk_step(c,1)&&nvm_ownership_declarations_type(c->owned->declarations,i,out);
}
static bool sdk_ref(SdkCheck *c,uint32_t i,uint32_t *out) {
    return sdk_step(c,1)&&nvm_sdk_provider_reference(c->owned->provider,i,out)&&*out<c->declarations.types;
}
static bool sdk_signature(SdkCheck *c,uint32_t i,NvmSdkSignatureRow *out) {
    return sdk_step(c,1)&&nvm_sdk_provider_signature(c->owned->provider,i,out);
}
static bool sdk_field_binding(SdkCheck *c,uint32_t layout,uint16_t field,uint32_t *type) {
    bool found=false;
    for(uint32_t i=0;i<c->counts[3];i++) {
        NvmSdkBindingRow b;if(!sdk_step(c,1)||!nvm_sdk_provider_binding(c->owned->provider,i,&b))return false;
        if(b.kind==NVM_SDK_BIND_FIELD&&b.subject==layout&&b.slot==field) {
            if(found)return false;
            *type=b.detail;found=true;
        }
    }
    return found;
}
/* A field's ARRAY binding names its element, while a provider field binding
 * names the complete ARRAY type. I compare that distinction explicitly. */
static bool sdk_array_element(SdkCheck *c,uint32_t layout,uint16_t field,uint32_t *type) {
    for(uint32_t i=0;i<c->declarations.bindings;i++) {
        NvmOrdinaryArrayBinding b;
        if(!sdk_step(c,1)||!nvm_ownership_declarations_binding(c->owned->declarations,i,&b))return false;
        if(b.layout==layout&&b.field==field){*type=b.element_type;return true;}
    }
    return false;
}
static bool sdk_equal(SdkCheck *,uint32_t,uint32_t);
static bool sdk_field_matches(SdkCheck *c,uint32_t layout,uint16_t field,uint32_t type) {
    NvmV2LayoutField f;NvmOrdinaryArrayType t;
    if(!nvm_ownership_declarations_field(c->owned->declarations,layout,field,&f)||!sdk_type(c,type,&t)||f.type_tag!=t.tag)return false;
    if(t.tag==TAG_ARRAY){uint32_t element;return sdk_array_element(c,layout,field,&element)&&(!c->identity_ready||sdk_equal(c,element,t.referent));}
    if(nvm_ownership_nominal_layout_kind(t.tag)>=0)return f.nested_idx==t.referent;
    return true;
}
static bool sdk_rows(SdkCheck *c) {
    for(unsigned i=0;i<256;i++)c->nominal[i]=UINT32_MAX;
    for(uint32_t i=0;i<c->counts[0];i++) {
        NvmSdkNominalRow n;if(!sdk_step(c,1)||!nvm_sdk_provider_nominal(c->owned->provider,i,&n))return false;
        if(!sdk_name(c,n.owner,true)||!sdk_name(c,n.name,false))return false;
        if(n.kind!=NVM_SDK_NOMINAL_OPAQUE) {
            NvmDeclarationLayout l;unsigned kind=n.kind==NVM_SDK_NOMINAL_RECORD?NVM_V2_LAYOUT_STRUCT:
                n.kind==NVM_SDK_NOMINAL_UNION?NVM_V2_LAYOUT_UNION:NVM_V2_LAYOUT_ENUM;
            if(!nvm_ownership_declarations_layout(c->owned->declarations,n.layout,&l)||l.kind!=kind)return false;
            if(c->nominal[n.layout]==UINT32_MAX)c->nominal[n.layout]=i;
        }
        for(uint32_t a=0;a<n.argument_count;a++){uint32_t type;if(!sdk_ref(c,n.argument_first+a,&type))return false;}
    }
    for(uint32_t i=0;i<c->declarations.types;i++) {
        NvmOrdinaryArrayType t;if(!sdk_type(c,i,&t))return false;
        if(t.tag==TAG_OPAQUE){NvmSdkNominalRow n;if(!nvm_sdk_provider_nominal(c->owned->provider,t.referent,&n)||n.kind!=NVM_SDK_NOMINAL_OPAQUE)return false;}
        else if(t.tag==TAG_FUNCTION){if(t.referent>=c->counts[2])return false;}
        else if(t.tag==TAG_STRUCT||t.tag==TAG_UNION||t.tag==TAG_ENUM){if(c->nominal[t.referent]==UINT32_MAX)return false;}
    }
    for(uint32_t i=0;i<c->declarations.layouts;i++) {
        const NvmV2Layout *l=&c->m->layouts.items[i];
        if(l->kind!=NVM_V2_LAYOUT_TUPLE&&c->nominal[i]==UINT32_MAX)return false;
        if(l->name_idx!=UINT32_MAX&&!sdk_name(c,l->name_idx,false))return false;
        for(uint16_t f=0;f<l->field_count;f++) {
            if(!sdk_step(c,1))return false;
            if(l->fields[f].name_idx!=UINT32_MAX&&!sdk_name(c,l->fields[f].name_idx,false))return false;
            if(l->fields[f].type_tag==TAG_FUNCTION||l->fields[f].type_tag==TAG_OPAQUE) {
                uint32_t type;if(!sdk_field_binding(c,i,f,&type)||!sdk_field_matches(c,i,f,type))return false;
            }
        }
    }
    for(uint32_t i=0;i<c->counts[2];i++) {
        NvmSdkSignatureRow s;if(!sdk_signature(c,i,&s)||s.coarse_signature>=c->m->signatures.count)return false;
        const NvmV2Signature *coarse=&c->m->signatures.items[s.coarse_signature];
        if(s.parameter_count!=coarse->param_count||s.result_count!=coarse->result_count)return false;
        for(unsigned result=0;result<2;result++) {
            uint32_t n=result?s.result_count:s.parameter_count,first=result?s.result_first:s.parameter_first;
            for(uint32_t j=0;j<n;j++) {
                uint32_t type;NvmOrdinaryArrayType t;
                if(!sdk_ref(c,first+j,&type)||!sdk_type(c,type,&t)||t.tag!=(result?coarse->result_tags[j]:coarse->param_tags[j]))return false;
            }
        }
    }
    for(uint32_t i=0;i<c->counts[1];i++) {
        NvmSdkProviderRow p;if(!sdk_step(c,1)||!nvm_sdk_provider_requirement(c->owned->provider,i,&p))return false;
        uint32_t strings[]={p.module,p.abi,p.target,p.artifact_digest,p.generation_digest,p.library};
        for(unsigned j=0;j<6;j++)if(!sdk_name(c,strings[j],false))return false;
    }
    for(uint32_t i=0;i<c->m->callbacks.count;i++) {
        const NvmV2Callback *cb=&c->m->callbacks.items[i];
        if(!sdk_step(c,1)||cb->import_idx>=c->m->imports.count||
           cb->abi_version!=NVM_CALLBACK_ABI_RETAINED_V1||
           (cb->execution!=NVM_FOREIGN_OWNER_THREAD&&cb->execution!=NVM_FOREIGN_WORKER_THREAD)||
           !sdk_name(c,cb->adapter_name_idx,false))return false;
        if(i) {
            const NvmV2Callback *prior=&c->m->callbacks.items[i-1];
            if(cb->import_idx<prior->import_idx||(cb->import_idx==prior->import_idx&&cb->parameter_idx<=prior->parameter_idx))return false;
        }
        if(cb->parameter_idx==NVM_CALLBACK_NO_PARAMETER) {
            if(cb->signature_idx!=UINT32_MAX)return false;
        } else {
            uint32_t selected=c->m->imports.items[cb->import_idx].signature_idx;
            if(selected>=c->m->signatures.count||cb->signature_idx>=c->m->signatures.count)return false;
            const NvmV2Signature *sig=&c->m->signatures.items[selected],*callback=&c->m->signatures.items[cb->signature_idx];
            if(cb->parameter_idx>=sig->param_count||sig->param_tags[cb->parameter_idx]!=TAG_FUNCTION||callback->result_count>1||
               !nvm_callback_shape_valid(callback->param_tags,callback->param_count,callback->result_count?callback->result_tags[0]:TAG_VOID))return false;
        }
    }
    for(uint32_t i=0;i<c->counts[3];i++) {
        NvmSdkBindingRow b;if(!sdk_step(c,1)||!nvm_sdk_provider_binding(c->owned->provider,i,&b))return false;
        for(uint32_t prior=0;prior<i;prior++) {
            NvmSdkBindingRow old;if(!sdk_step(c,1)||!nvm_sdk_provider_binding(c->owned->provider,prior,&old))return false;
            if(old.kind==b.kind&&old.subject==b.subject&&old.slot==b.slot)return false;
        }
        if(b.kind==NVM_SDK_BIND_FIELD) {
            if(!sdk_field_matches(c,b.subject,(uint16_t)b.slot,b.detail))return false;
        } else {
            NvmSdkSignatureRow sig;if(!sdk_signature(c,b.detail,&sig))return false;
            if(b.kind==NVM_SDK_BIND_FUNCTION) {
                if(b.subject>=c->m->functions.count||sig.coarse_signature!=c->m->functions.items[b.subject].signature_idx)return false;
            } else {
                if(b.subject>=c->m->imports.count||sig.coarse_signature!=c->m->imports.items[b.subject].signature_idx)return false;
                const NvmV2Import *import=&c->m->imports.items[b.subject];
                if(import->kind>NVM_V2_IMPORT_KIND_MAX||!sdk_name(c,import->module_name_idx,false)||!sdk_name(c,import->symbol_name_idx,false))return false;
            }
        }
    }
    /* I require exact details for every function/import in this new private
     * profile, including scalar-only subjects; old modules keep old entry points. */
    for(unsigned kind=0;kind<2;kind++) {
        uint32_t count=kind?c->m->functions.count:c->m->imports.count;
        for(uint32_t subject=0;subject<count;subject++) {
            bool found=false;
            for(uint32_t i=0;i<c->counts[3];i++) {
                NvmSdkBindingRow b;if(!sdk_step(c,1)||!nvm_sdk_provider_binding(c->owned->provider,i,&b))return false;
                if(b.kind==kind&&b.subject==subject)found=true;
            }
            if(!found)return false;
        }
    }
    return true;
}
/* Nodes share existing namespaces. Only ARRAY, FUNCTION and OPAQUE edges may
 * guard a reference cycle; storage and generic-key edges never grant that. */
static uint32_t sdk_layout_base(SdkCheck *c){return c->declarations.types;}
static uint32_t sdk_signature_base(SdkCheck *c){return sdk_layout_base(c)+c->declarations.layouts;}
static uint32_t sdk_nominal_base(SdkCheck *c){return sdk_signature_base(c)+c->counts[2];}
static uint32_t sdk_graph_count(SdkCheck *c){return sdk_nominal_base(c)+c->counts[0];}
static bool sdk_edge(SdkCheck *c,uint32_t node,uint32_t next,uint32_t *child,bool *guard,bool *end) {
    *child=UINT32_MAX;*guard=false;*end=false;if(!sdk_step(c,1))return false;
    uint32_t lb=sdk_layout_base(c),sb=sdk_signature_base(c),nb=sdk_nominal_base(c);
    if(node<lb) {
        NvmOrdinaryArrayType t;if(!sdk_type(c,node,&t))return false;
        if(t.tag==TAG_ARRAY||t.tag==TAG_FUNCTION||t.tag==TAG_OPAQUE) {
            if(next){*end=true;return true;}*guard=true;
            *child=t.tag==TAG_ARRAY?t.referent:t.tag==TAG_FUNCTION?sb+t.referent:nb+t.referent;
        } else if(nvm_ownership_nominal_layout_kind(t.tag)>=0) {
            if(next==0)*child=lb+t.referent;
            else if(next==1&&t.tag!=TAG_TUPLE)*child=nb+c->nominal[t.referent];
            else *end=true;
        } else *end=true;
    } else if(node<sb) {
        uint32_t index=node-lb;const NvmV2Layout *l=&c->m->layouts.items[index];
        if(next>=l->field_count) {
            if(next==l->field_count&&l->kind!=NVM_V2_LAYOUT_TUPLE)*child=nb+c->nominal[index];
            else *end=true;
            return true;
        }
        const NvmV2LayoutField *f=&l->fields[next];
        if(nvm_ownership_nominal_layout_kind(f->type_tag)>=0)*child=lb+f->nested_idx;
        else if(f->type_tag==TAG_ARRAY){*guard=true;if(!sdk_array_element(c,index,(uint16_t)next,child))return false;}
        else if(f->type_tag==TAG_FUNCTION||f->type_tag==TAG_OPAQUE) {
            if(!sdk_field_binding(c,index,(uint16_t)next,child))return false;
        }
    } else if(node<nb) {
        NvmSdkSignatureRow s;if(!sdk_signature(c,node-sb,&s))return false;
        if(next>=s.parameter_count+s.result_count){*end=true;return true;}
        if(!sdk_ref(c,next<s.parameter_count?s.parameter_first+next:s.result_first+next-s.parameter_count,child))return false;
    } else {
        NvmSdkNominalRow n;if(!nvm_sdk_provider_nominal(c->owned->provider,node-nb,&n))return false;
        if(next>=n.argument_count){*end=true;return true;}
        if(!sdk_ref(c,n.argument_first+next,child))return false;
    }
    return *end||*child==UINT32_MAX||*child<sdk_graph_count(c);
}
static void *sdk_allocate(SdkCheck *c,size_t n,size_t width) {
    if(width&&n>SIZE_MAX/width){c->error=NVM_SDK_LIMIT;return NULL;}
    if(!n)n=1;
    size_t bytes=n*width;
    if(!nvm_preparation_charge(c->budget,bytes,0)){c->error=NVM_SDK_LIMIT;return NULL;}
    void *p=calloc(n,width);if(!p)c->error=NVM_SDK_MEMORY;return p;
}
static bool sdk_graph(SdkCheck *c) {
    typedef struct {uint32_t node,next;} Frame;
    uint32_t count=sdk_graph_count(c);if(!count)return true;
    uint8_t *color=sdk_allocate(c,count,1);
    Frame *stack=sdk_allocate(c,count,sizeof *stack);bool ok=false;
    if(!color||!stack)goto done;
    for(uint32_t root=0;root<count;root++) {
        if(color[root])continue;
        uint32_t depth=1;stack[0]=(Frame){root,0};color[root]=1;
        while(depth) {
            Frame *f=&stack[depth-1];uint32_t child;bool guard,end;
            if(!sdk_edge(c,f->node,f->next++,&child,&guard,&end))goto done;
            if(end){color[f->node]=2;depth--;continue;}
            /* I remove only explicit reference edges. The remaining whole
             * graph must be acyclic, regardless of DFS traversal order. */
            if(child==UINT32_MAX||guard)continue;
            if(color[child]==1)goto done;
            if(color[child]==2)continue;
            if(depth>=count)goto done;
            color[child]=1;stack[depth++]=(Frame){child,0};
        }
    }
    ok=true;
done:free(color);free(stack);return ok;
}
/* Equality is coinductive only after sdk_graph established the permitted
 * reference cycles. Nominal equality compares keys, not C names or layouts. */
typedef struct {uint32_t left,right;} SdkPair;
static bool sdk_equal(SdkCheck *c,uint32_t left,uint32_t right) {
    const uint32_t maximum=65536;
    if(!c->pairs)c->pairs=sdk_allocate(c,maximum,sizeof(SdkPair));
    SdkPair *pairs=c->pairs;if(!pairs)return false;
    uint32_t count=1,at=0;pairs[0]=(SdkPair){left,right};bool equal=false;
    uint32_t lb=sdk_layout_base(c),sb=sdk_signature_base(c),nb=sdk_nominal_base(c);
    while(at<count) {
        SdkPair p=pairs[at++];if(!sdk_step(c,1))goto done;
        if(p.left==p.right)continue;
        unsigned a=p.left<lb?0:p.left<sb?1:p.left<nb?2:3;
        unsigned b=p.right<lb?0:p.right<sb?1:p.right<nb?2:3;if(a!=b)goto done;
        if(a==0) {
            NvmOrdinaryArrayType x,y;if(!sdk_type(c,p.left,&x)||!sdk_type(c,p.right,&y)||x.tag!=y.tag)goto done;
            /* Named types compare their complete declaration keys only. */
            if(x.tag==TAG_STRUCT||x.tag==TAG_UNION||x.tag==TAG_ENUM) {
                p.left=nb+c->nominal[x.referent];p.right=nb+c->nominal[y.referent];a=3;
            }
        }
        if(a==1 && c->m->layouts.items[p.left-lb].kind!=NVM_V2_LAYOUT_TUPLE) {
            if(c->m->layouts.items[p.right-lb].kind==NVM_V2_LAYOUT_TUPLE)goto done;
            p.left=nb+c->nominal[p.left-lb];p.right=nb+c->nominal[p.right-lb];a=3;
        }
        if(a==1) {
            const NvmV2Layout *x=&c->m->layouts.items[p.left-lb],*y=&c->m->layouts.items[p.right-lb];
            if(x->kind!=y->kind||x->field_count!=y->field_count)goto done;
            for(uint16_t i=0;i<x->field_count;i++)if(!sdk_step(c,1)||x->fields[i].type_tag!=y->fields[i].type_tag)goto done;
        } else if(a==2) {
            NvmSdkSignatureRow x,y;if(!sdk_signature(c,p.left-sb,&x)||!sdk_signature(c,p.right-sb,&y)||
                x.parameter_count!=y.parameter_count||x.result_count!=y.result_count)goto done;
        } else if(a==3) {
            NvmSdkNominalRow x,y;
            if(!nvm_sdk_provider_nominal(c->owned->provider,p.left-nb,&x)||!nvm_sdk_provider_nominal(c->owned->provider,p.right-nb,&y)||
               x.kind!=y.kind||x.argument_count!=y.argument_count||!sdk_same_name(c,x.owner,y.owner)||!sdk_same_name(c,x.name,y.name))goto done;
        }
        for(uint32_t edge=0;;edge++) {
            uint32_t x,y;bool gx,gy,ex,ey;
            if(!sdk_edge(c,p.left,edge,&x,&gx,&ex)||!sdk_edge(c,p.right,edge,&y,&gy,&ey)||ex!=ey)goto done;
            if(ex)break;
            if(gx!=gy || (x==UINT32_MAX)!=(y==UINT32_MAX))goto done;
            if(x==UINT32_MAX||x==y)continue;
            bool seen=false;
            for(uint32_t i=0;i<count;i++){if(!sdk_step(c,1))goto done;if(pairs[i].left==x&&pairs[i].right==y){seen=true;break;}}
            if(!seen){if(count==maximum){c->error=NVM_SDK_LIMIT;goto done;}pairs[count++]=(SdkPair){x,y};}
        }
    }
    equal=true;
done:return equal;
}
static bool sdk_nominal_keys(SdkCheck *c) {
    for(uint32_t i=0;i<c->counts[0];i++) {
        NvmSdkNominalRow n;if(!nvm_sdk_provider_nominal(c->owned->provider,i,&n))return false;
        for(uint32_t j=0;j<i;j++) {
            NvmSdkNominalRow old;if(!sdk_step(c,1)||!nvm_sdk_provider_nominal(c->owned->provider,j,&old))return false;
            bool same=sdk_equal(c,sdk_nominal_base(c)+i,sdk_nominal_base(c)+j);
            if(c->error!=NVM_SDK_INVALID)return false;
            if(same&&n.layout!=old.layout)return false;
            if(n.layout!=UINT32_MAX&&n.layout==old.layout&&!same)return false;
        }
    }
    return true;
}
static bool sdk_complete_fields(SdkCheck *c) {
    c->identity_ready=true;
    for(uint32_t i=0;i<c->counts[3];i++) {
        NvmSdkBindingRow b;if(!sdk_step(c,1)||!nvm_sdk_provider_binding(c->owned->provider,i,&b))return false;
        if(b.kind==NVM_SDK_BIND_FIELD&&!sdk_field_matches(c,b.subject,(uint16_t)b.slot,b.detail))return false;
    }
    return true;
}
static bool sdk_lifetime_nodes(SdkCheck *c) {
    for(uint32_t i=0;i<c->lifetimes[1];i++) {
        NvmSdkLifetimeNode n;NvmOrdinaryArrayType t;
        if(!sdk_step(c,1)||!nvm_sdk_provider_lifetime_node(c->owned->provider,i,&n)||!sdk_type(c,n.type,&t))return false;
        if(n.hook_set!=UINT32_MAX&&!sdk_name(c,n.hook_set,false))return false;
        uint32_t children=0;
        if(t.tag==TAG_ARRAY)children=1;
        else if(t.tag==TAG_FUNCTION) {
            NvmSdkSignatureRow s;if(!sdk_signature(c,t.referent,&s))return false;
            children=s.parameter_count+s.result_count;
            const NvmV2Signature *coarse=&c->m->signatures.items[s.coarse_signature];
            if((n.mode!=NVM_SDK_CALLBACK_CALL&&n.mode!=NVM_SDK_CALLBACK_RETAINED)||
               coarse->result_count>1||!nvm_callback_shape_valid(coarse->param_tags,coarse->param_count,
                   coarse->result_count?coarse->result_tags[0]:TAG_VOID))return false;
        } else if(t.tag==TAG_STRUCT||t.tag==TAG_TUPLE||t.tag==TAG_UNION)
            children=c->m->layouts.items[t.referent].field_count;
        if(n.child_count!=children)return false;
        bool primitive=t.tag==TAG_INT||t.tag==TAG_U8||t.tag==TAG_BOOL||t.tag==TAG_FLOAT||t.tag==TAG_ENUM;
        if(primitive!=(n.mode==NVM_SDK_VALUE))return false;
        if(t.tag==TAG_OPAQUE) {
            if(n.mode!=NVM_SDK_OPAQUE_PIN&&n.mode!=NVM_SDK_OPAQUE_PROVIDER_LIFETIME&&n.mode!=NVM_SDK_BORROW_ARGUMENT_RESULT)return false;
        } else if(n.mode==NVM_SDK_OPAQUE_PIN||n.mode==NVM_SDK_OPAQUE_PROVIDER_LIFETIME)return false;
        if((t.tag==TAG_FUNCTION)!=(n.mode==NVM_SDK_CALLBACK_CALL||n.mode==NVM_SDK_CALLBACK_RETAINED))return false;
        if((n.mode==NVM_SDK_OPAQUE_PIN||n.mode==NVM_SDK_CALLBACK_CALL||n.mode==NVM_SDK_CALLBACK_RETAINED)&&n.hook_set==UINT32_MAX)return false;
        if((n.mode==NVM_SDK_VALUE||n.mode==NVM_SDK_BORROW_CALL||n.mode==NVM_SDK_BORROW_MUTABLE_CALL||
            n.mode==NVM_SDK_BORROW_ARGUMENT_RESULT||n.mode==NVM_SDK_OPAQUE_PROVIDER_LIFETIME)&&n.hook_set!=UINT32_MAX)return false;
        if(n.mode!=NVM_SDK_BORROW_ARGUMENT_RESULT&&n.owner_argument!=UINT32_MAX)return false;
        for(uint32_t j=0;j<children;j++) {
            NvmSdkLifetimeNode child;if(!sdk_step(c,1)||!nvm_sdk_provider_lifetime_node(c->owned->provider,n.child_first+j,&child))return false;
            if(t.tag==TAG_ARRAY){if(!sdk_equal(c,t.referent,child.type))return false;}
            else if(t.tag==TAG_FUNCTION) {
                NvmSdkSignatureRow sig;uint32_t type;
                if(!sdk_signature(c,t.referent,&sig)||!sdk_ref(c,j<sig.parameter_count?sig.parameter_first+j:sig.result_first+j-sig.parameter_count,&type)||!sdk_equal(c,type,child.type))return false;
            } else if(!sdk_field_matches(c,t.referent,(uint16_t)j,child.type))return false;
            /* A container drop cannot independently drop the same child.
             * Independent child ownership needs a future exact schema proof. */
            if(n.hook_set!=UINT32_MAX&&child.hook_set!=UINT32_MAX)return false;
        }
    }
    return true;
}
/* I permit policy back edges only across explicit ARRAY/FUNCTION references.
 * Type agreement alone is not used as permission for a by-value policy cycle. */
static bool sdk_policy_graph(SdkCheck *c) {
    typedef struct {uint32_t node,next;} Frame;
    uint32_t count=c->lifetimes[1];if(!count)return true;
    uint8_t *color=sdk_allocate(c,count,1);
    Frame *stack=sdk_allocate(c,count,sizeof *stack);bool ok=false;
    if(!color||!stack)goto done;
    for(uint32_t root=0;root<count;root++) {
        if(color[root])continue;
        uint32_t depth=1;stack[0]=(Frame){root,0};color[root]=1;
        while(depth) {
            Frame *f=&stack[depth-1];NvmSdkLifetimeNode n;NvmOrdinaryArrayType t;
            if(!sdk_step(c,1)||!nvm_sdk_provider_lifetime_node(c->owned->provider,f->node,&n)||!sdk_type(c,n.type,&t))goto done;
            if(f->next==n.child_count||t.tag==TAG_ARRAY||t.tag==TAG_FUNCTION) {
                color[f->node]=2;depth--;continue;
            }
            uint32_t child=n.child_first+f->next++;
            if(color[child]==1)goto done;
            if(color[child]==2)continue;
            if(depth>=count)goto done;
            color[child]=1;stack[depth++]=(Frame){child,0};
        }
    }
    ok=true;
done:free(color);free(stack);return ok;
}
typedef struct {uint32_t node,arguments;bool result;} SdkPolicyVisit;
static bool sdk_policy_context(SdkCheck *c,NvmSdkCallPolicy policy,SdkPolicyVisit *queue,uint8_t *reached) {
    uint32_t count=0,at=0;const uint32_t maximum=65536;
    for(unsigned result=0;result<2;result++) {
        uint32_t n=result?policy.result_count:policy.parameter_count;
        uint32_t first=result?policy.result_first:policy.parameter_first;
        if(n>maximum-count){c->error=NVM_SDK_LIMIT;return false;}
        for(uint32_t i=0;i<n;i++)queue[count++]=(SdkPolicyVisit){first+i,policy.parameter_count,result!=0};
    }
    while(at<count) {
        SdkPolicyVisit visit=queue[at++];NvmSdkLifetimeNode n;NvmOrdinaryArrayType type;
        if(!sdk_step(c,1)||!nvm_sdk_provider_lifetime_node(c->owned->provider,visit.node,&n)||!sdk_type(c,n.type,&type))return false;
        reached[visit.node]=1;
        if((n.mode==NVM_SDK_BORROW_CALL||n.mode==NVM_SDK_BORROW_MUTABLE_CALL)&&visit.result)return false;
        if((n.mode==NVM_SDK_SNAPSHOT_RESULT||n.mode==NVM_SDK_BORROW_ARGUMENT_RESULT)&&!visit.result)return false;
        if(n.mode==NVM_SDK_BORROW_ARGUMENT_RESULT&&n.owner_argument>=visit.arguments)return false;
        NvmSdkSignatureRow signature={0};
        if(type.tag==TAG_FUNCTION&&!sdk_signature(c,type.referent,&signature))return false;
        for(uint32_t i=0;i<n.child_count;i++) {
            SdkPolicyVisit child={n.child_first+i,visit.arguments,visit.result};
            if(type.tag==TAG_FUNCTION){child.arguments=signature.parameter_count;child.result=i<signature.parameter_count?!visit.result:visit.result;}
            bool seen=false;
            for(uint32_t j=0;j<count;j++) {
                if(!sdk_step(c,1))return false;
                if(queue[j].node==child.node&&queue[j].arguments==child.arguments&&queue[j].result==child.result){seen=true;break;}
            }
            if(!seen){if(count==maximum){c->error=NVM_SDK_LIMIT;return false;}queue[count++]=child;}
        }
    }
    return true;
}
static bool sdk_policies(SdkCheck *c) {
    if(!sdk_lifetime_nodes(c)||!sdk_policy_graph(c))return false;
    if(!c->lifetimes[1]&&!c->lifetimes[0])return true;
    SdkPolicyVisit *queue=sdk_allocate(c,65536,sizeof *queue);
    uint8_t *reached=sdk_allocate(c,c->lifetimes[1]?c->lifetimes[1]:1,1);bool ok=false;
    if(!queue||!reached)goto done;
    for(uint32_t i=0;i<c->lifetimes[0];i++) {
        NvmSdkCallPolicy policy;if(!sdk_step(c,1)||!nvm_sdk_provider_call_policy(c->owned->provider,i,&policy))goto done;
        if(policy.execution!=UINT32_MAX) {
            if(policy.execution>=c->m->callbacks.count||c->m->callbacks.items[policy.execution].parameter_idx!=NVM_CALLBACK_NO_PARAMETER)goto done;
        }
        if(!sdk_policy_context(c,policy,queue,reached))goto done;
    }
    for(uint32_t i=0;i<c->lifetimes[1];i++)if(!reached[i]) {
        NvmSdkLifetimeNode n;if(!sdk_step(c,1)||!nvm_sdk_provider_lifetime_node(c->owned->provider,i,&n))goto done;
        if(n.owner_argument!=UINT32_MAX||n.callback_profile)goto done;
    }
    for(uint32_t i=0;i<c->counts[3];i++) {
        NvmSdkBindingRow b;if(!sdk_step(c,1)||!nvm_sdk_provider_binding(c->owned->provider,i,&b))goto done;
        if(b.kind!=NVM_SDK_BIND_IMPORT)continue;
        uint32_t selector;NvmSdkCallPolicy policy;NvmSdkSignatureRow signature;
        if(!nvm_sdk_provider_binding_policy(c->owned->provider,i,&selector)||
           !nvm_sdk_provider_call_policy(c->owned->provider,selector,&policy)||!sdk_signature(c,b.detail,&signature)||
           policy.parameter_count!=signature.parameter_count||policy.result_count!=signature.result_count)goto done;
        if(policy.execution!=UINT32_MAX&&c->m->callbacks.items[policy.execution].import_idx!=b.subject)goto done;
        for(unsigned result=0;result<2;result++) {
            uint32_t n=result?policy.result_count:policy.parameter_count,first=result?policy.result_first:policy.parameter_first;
            for(uint32_t j=0;j<n;j++) {
                NvmSdkLifetimeNode node;uint32_t type;
                if(!sdk_step(c,1)||!nvm_sdk_provider_lifetime_node(c->owned->provider,first+j,&node)||
                   !sdk_ref(c,result?signature.result_first+j:signature.parameter_first+j,&type)||!sdk_equal(c,type,node.type))goto done;
                NvmOrdinaryArrayType t;if(!sdk_type(c,type,&t))goto done;
                if(!result&&t.tag==TAG_FUNCTION) {
                    bool found=false;NvmSdkSignatureRow callback_sig;if(!sdk_signature(c,t.referent,&callback_sig))goto done;
                    for(uint32_t k=0;k<c->m->callbacks.count;k++) {
                        if(!sdk_step(c,1))goto done;
                        const NvmV2Callback *cb=&c->m->callbacks.items[k];
                        if(cb->import_idx==b.subject&&cb->parameter_idx==j) {
                            if(found||cb->signature_idx!=callback_sig.coarse_signature||
                               node.callback_profile!=((uint32_t)cb->abi_version|((uint32_t)cb->execution<<8)))goto done;
                            found=true;
                        }
                    }
                    if(!found)goto done;
                }
            }
        }
    }
    ok=true;
done:free(queue);free(reached);return ok;
}
void nvm_sdk_description_free(NvmSdkDescription *p) {
    if(!p)return;
    nvm_sdk_provider_transport_free(p->provider);
    nvm_ownership_declarations_free(p->declarations);
    nvm_sdk_module_snapshot_free(p->module);free(p);
}
const NvmV2Module *nvm_sdk_description_module(const NvmSdkDescription *p) {
    return p?nvm_sdk_module_snapshot_view(p->module):NULL;
}
NvmSdkResult nvm_sdk_description_prepare(const NvmV2Module *module,const uint8_t *wire,size_t bytes,
        NvmPreparationBudget *budget,NvmSdkDescription **out) {
    if(!module||!budget||!out)return NVM_SDK_INVALID;
    if(!nvm_preparation_budget_valid(budget))return NVM_SDK_LIMIT;
    NvmPreparationBudget remaining=*budget;
    if(!nvm_preparation_charge(&remaining,sizeof(NvmSdkDescription),0))return NVM_SDK_LIMIT;
    NvmSdkDescription *p=calloc(1,sizeof *p);if(!p)return NVM_SDK_MEMORY;
    NvmSdkResult result=nvm_sdk_module_snapshot_prepare_budget(module,&remaining,&p->module);
    if(result!=NVM_SDK_OK)goto done;
    const NvmV2Module *owned=nvm_sdk_module_snapshot_view(p->module);
    NvmDeclarationResult decl=nvm_prepare_ownership_declarations_typed_v2(owned,&remaining,&p->declarations);
    if(decl.status!=NVM_DECL_PREPARED) {
        result=decl.status==NVM_DECL_LIMIT?NVM_SDK_LIMIT:decl.status==NVM_DECL_MEMORY?NVM_SDK_MEMORY:NVM_SDK_INVALID;goto done;
    }
    result=nvm_sdk_provider_lifetime_decode_budget(wire,bytes,&remaining,&p->provider);if(result!=NVM_SDK_OK)goto done;
    SdkCheck c={0};c.owned=p;c.m=owned;c.budget=&remaining;c.error=NVM_SDK_INVALID;
    if(!nvm_ownership_declarations_counts(p->declarations,&c.declarations)||
       !nvm_sdk_provider_counts(p->provider,c.counts)||!nvm_sdk_provider_lifetime_counts(p->provider,c.lifetimes)) {result=NVM_SDK_INVALID;goto done;}
    bool valid=sdk_rows(&c)&&sdk_graph(&c)&&sdk_nominal_keys(&c)&&sdk_complete_fields(&c)&&sdk_policies(&c);
    free(c.pairs);if(!valid){result=c.error;goto done;}
    *out=p;*budget=remaining;return NVM_SDK_OK;
done:nvm_sdk_description_free(p);return result;
}
