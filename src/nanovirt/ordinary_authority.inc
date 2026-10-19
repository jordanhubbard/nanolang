/* I publish complete ordinary declarations, independently of instruction flow. */
#include "../nanoisa/ownership_contracts.h"
#include "../nanoisa/retained_layouts.h"
#include "../nanoisa/record_array_admission.h"
#include "../nanoisa/verifier.h"
static void ordinary_word(uint8_t *out,uint32_t value,unsigned width) {
    for(unsigned i=0;i<width;i++)out[i]=(uint8_t)(value>>(i*8));
}
/* I omit optional facts for cyclic declarations without rejecting source. */
static bool ordinary_graph(const NvmV2Layouts *layouts) {
    if (layouts->count > MAX_STRUCT_DEFS) return false;
    uint8_t colors[MAX_STRUCT_DEFS] = {0};
    uint32_t stack[MAX_STRUCT_DEFS], next[MAX_STRUCT_DEFS] = {0};
    for (uint32_t root = 0; root < layouts->count; root++) {
        if (colors[root]) continue;
        unsigned depth = 1; stack[0] = root; colors[root] = 1;
        while (depth) {
            uint32_t current = stack[depth - 1];
            const NvmV2Layout *record = &layouts->items[current];
            if (next[current] == record->field_count) {
                colors[current] = 2; depth--; continue;
            }
            uint32_t child = record->fields[next[current]++].nested_idx;
            if (child == NVM_V2_NO_INDEX) continue;
            if (child >= layouts->count || colors[child] == 1) return false;
            if (colors[child] == 2) continue;
            colors[child] = 1; stack[depth++] = child;
        }
    }
    return true;
}
/* I cover loaded declarations by identity, not by source spelling. */
static bool ordinary_declarations(CG *cg, ASTNode *program, ModuleList *modules,
                                  ASTNode **records) {
    bool seen[MAX_STRUCT_DEFS]={0};
    unsigned count=cg->struct_count;
    int imported=modules?modules->count:0;
    for(int source=0;source<=imported;source++) {
        ASTNode *tree=source==imported?program:get_cached_module_ast(modules->module_paths[source]);
        if(!tree || tree->type!=AST_PROGRAM)return false;
        for(int i=0;i<tree->as.program.count;i++) {
            ASTNode *node=bytecode_declaration(tree->as.program.items[i]);
            if(node->type==AST_ENUM_DEF || node->type==AST_UNION_DEF)return false;
            if(node->type==AST_IMPORT && !imported)return false;
            if(node->type!=AST_STRUCT_DEF)continue;
            unsigned ordinal=0;
            while(ordinal<count && cg->structs[ordinal].declaration!=node)ordinal++;
            if(ordinal==count)return false;
            seen[ordinal]=true;
        }
    }
    for(unsigned i=0;i<count;i++) {
        const CgStructDef *registered=&cg->structs[i];
        ASTNode *node=registered->declaration;
        if(!seen[i] || !node || registered->declaration_ambiguous || node->type!=AST_STRUCT_DEF ||
           registered->def_idx!=i || registered->name!=node->as.struct_def.name ||
           node->as.struct_def.is_resource || node->as.struct_def.is_extern ||
           strchr(node->as.struct_def.name,'<'))return false;
        for(unsigned k=0;k<i;k++)
            if(!strcmp(node->as.struct_def.name,records[k]->as.struct_def.name))return false;
        records[i]=node;
    }
    return true;
}

static void publish_ordinary_authority(CG *cg,ASTNode *program,ModuleList *modules) {
    NvmModule *module=cg->module;
    if(!cg->struct_count || cg->enum_count || cg->union_count || cg->extern_count || cg->passive ||
       module->layout_size || module->ownership_size)return;
    ASTNode *records[MAX_STRUCT_DEFS]={0};unsigned count=cg->struct_count;
    if(!ordinary_declarations(cg,program,modules,records))return;
    size_t size=8+((count+3u)&~3u)+4;
    for(uint32_t f=0;f<module->function_count;f++) {
        NvmFunctionEntry *fn=&module->functions[f];
        if(fn->upvalue_count || fn->arity>fn->local_count || fn->result_count>1)return;
        size_t part=4+((size_t)fn->local_count+1)*8;
        if(size>UINT32_MAX-part)return;
        size+=part;
    }
    uint8_t *data=calloc(size,1);
    NvmV2Layout *items=calloc(count,sizeof *items);
    if(!data || !items){free(data);free(items);cg_error(cg,0,"I cannot allocate ordinary declaration facts");return;}
    NvmV2Layouts layouts={items,count};
    bool supported=true;
    uint32_t arrays=0;uint8_t elements[5]={0};
    for(unsigned i=0;i<count && supported;i++) {
        ASTNode *node=records[i];int fields=node->as.struct_def.field_count;
        if(fields<0 || fields>UINT16_MAX){supported=false;break;}
        items[i]=(NvmV2Layout){NVM_V2_LAYOUT_STRUCT,(uint16_t)fields,NVM_V2_NO_INDEX,NULL};
        items[i].fields=fields?calloc((size_t)fields,sizeof *items[i].fields):NULL;
        if(fields && !items[i].fields){cg_error(cg,0,"I cannot allocate ordinary record fields");supported=false;break;}
        for(int j=0;j<fields;j++) {
            TypeInfo *info=node->as.struct_def.field_type_info?node->as.struct_def.field_type_info[j]:NULL;
            if(info && (info->type_param_count || (info->element_type && info->base_type!=TYPE_ARRAY))){supported=false;break;}
            Type declared=info?info->base_type:node->as.struct_def.field_types[j];
            uint8_t tag=ordinary_slot_tag(declared);uint32_t child=NVM_V2_NO_INDEX;
            if(tag==TAG_ARRAY) {
                TypeInfo *element=info?info->element_type:NULL;
                uint8_t elem=element?ordinary_slot_tag(element->base_type):TAG_COUNT;
                if(!element || elem<TAG_INT || elem>TAG_STRING || element->element_type ||
                   element->type_param_count || element->service_declaration || element->generic_name ||
                   (node->as.struct_def.field_element_types && node->as.struct_def.field_element_types[j]!=element->base_type)) {supported=false;break;}
                elements[elem-1]=1;arrays++;
            }
            if(tag==TAG_STRUCT) {
                const char *name=node->as.struct_def.field_type_names?node->as.struct_def.field_type_names[j]:NULL;
                if(name)for(unsigned k=0;k<count;k++)if(!strcmp(name,records[k]->as.struct_def.name))child=k;
                if(child==NVM_V2_NO_INDEX){supported=false;break;}
            }
            if(tag==TAG_COUNT){supported=false;break;}
            items[i].fields[j]=(NvmV2LayoutField){tag,child,NVM_V2_NO_INDEX};
        }
    }
    if (supported && !ordinary_graph(&layouts)) supported=false;
    ordinary_word(data,1,4);ordinary_word(data+4,count,4);
    memset(data+8,NVM_LAYOUT_COMPLETE,count);
    size_t at=8+((count+3u)&~3u);ordinary_word(data+at,module->function_count,4);at+=4;
    for(uint32_t f=0;f<module->function_count && supported;f++) {
        NvmFunctionEntry *fn=&module->functions[f];
        ordinary_word(data+at,fn->local_count,2);ordinary_word(data+at+2,fn->arity,2);at+=4;
        data[at]=fn->result_count?fn->result_tag:TAG_VOID;
        ordinary_word(data+at+4,NVM_V2_NO_INDEX,4);at+=8;
        for(uint16_t local=0;local<fn->local_count;local++) {
            uint8_t tag=TAG_COUNT;bool conflict=false;
            for(CgAuthoritySlot *fact=*cg->authority_slots;fact;fact=fact->next) {
                if(fact->function!=f || fact->slot!=local)continue;
                if(tag!=TAG_COUNT && tag!=fact->tag)conflict=true;
                if(fact->tag==TAG_COUNT)conflict=true;
                tag=fact->tag;
            }
            if(local<fn->arity) {
                if(!module->function_param_types || !module->function_param_types[f] ||
                   module->function_param_types[f][local]!=tag)conflict=true;
            }
            if(conflict || tag==TAG_COUNT || (tag==TAG_ARRAY && !arrays)){supported=false;break;}
            data[at]=tag;ordinary_word(data+at+4,NVM_V2_NO_INDEX,4);at+=8;
        }
    }
    if(supported && arrays) {
        uint32_t types=0,index[5]={0};
        for(unsigned i=0;i<5;i++)if(elements[i])index[i]=types++;
        size_t extra=28u+(size_t)types*8u+(size_t)arrays*12u;
        if(extra>UINT32_MAX-size){supported=false;}
        else {
            uint8_t *grown=realloc(data,size+extra);
            if(!grown){cg_error(cg,0,"I cannot allocate flat-array declaration facts");supported=false;}
            else {
                data=grown;memset(data+size,0,extra);ordinary_word(data,3,4);at=size;
                ordinary_word(data+at,4,4);at+=8; /* Empty bounded path table. */
                ordinary_word(data+at,1,4);at+=4;
                ordinary_word(data+at,2,2);ordinary_word(data+at+2,1,2);
                ordinary_word(data+at+4,(uint32_t)(8u+types*8u+arrays*12u),4);at+=8;
                ordinary_word(data+at,types,4);at+=4;
                for(unsigned i=0;i<5;i++)if(elements[i]) {
                    data[at]=(uint8_t)(i+1);ordinary_word(data+at+4,NVM_V2_NO_INDEX,4);at+=8;
                }
                ordinary_word(data+at,arrays,4);at+=4;
                for(unsigned i=0;i<count;i++)for(unsigned j=0;j<items[i].field_count;j++)
                    if(items[i].fields[j].type_tag==TAG_ARRAY) {
                        uint8_t elem=ordinary_slot_tag(records[i]->as.struct_def.field_type_info[j]->element_type->base_type);
                        ordinary_word(data+at,i,4);ordinary_word(data+at+4,j,2);
                        ordinary_word(data+at+8,index[elem-1],4);at+=12;
                    }
                size+=extra;
            }
        }
    }
    if(supported) {
        /* My shallow staged view owns only the two new optional buffers. */
        NvmModule staged=*module;staged.layout_data=NULL;staged.layout_size=0;
        staged.ownership_data=data;staged.ownership_size=(uint32_t)size;
        NvmV2Result result=NVM_V2_OK;bool needs=false;
        if(arrays) {
            /* My legacy retained decoder excludes mixed forward array DAGs.
             * These private bytes acquire authority only after the selected proof. */
            size_t layout_bytes=nvm_v2_layouts_encoded_size(&layouts);
            if(layout_bytes>UINT32_MAX)result=NVM_V2_ERR_INDEX_RANGE;
            else {
                staged.layout_data=malloc(layout_bytes);
                if(!staged.layout_data)result=NVM_V2_ERR_TRUNCATED;
                else {
                    staged.layout_size=(uint32_t)layout_bytes;
                    result=nvm_v2_layouts_encode(&layouts,staged.layout_data,layout_bytes);
                }
            }
        } else result=nvm_retain_layouts(&staged,&layouts);
        bool eligible=false,omitted=false;
        if(result==NVM_V2_OK && arrays) {
            NvmRecordArrayExecutionPlan *plan=NULL;
            NvmArrayEligibilityResult admission=nvm_record_array_admit(&staged,&plan);
            nvm_record_array_execution_free(plan);
            omitted=admission.status==NVM_ARRAY_UNRESOLVED || admission.status==NVM_ARRAY_LIMIT;
            if(admission.status==NVM_ARRAY_ELIGIBLE)eligible=nvm_verify(&staged).ok;
        } else if(result==NVM_V2_OK) {
            result=nvm_ownership_contracts_validate(&staged,&needs);
            eligible=result==NVM_V2_OK && !needs;
        }
        if(eligible) {
            module->layout_data=staged.layout_data;module->layout_size=staged.layout_size;
            module->ownership_data=data;module->ownership_size=(uint32_t)size;data=NULL;
        } else {free(staged.layout_data);if(!omitted)cg_error(cg,0,"I cannot validate ordinary declaration publication");}
    }
    free(data);nvm_v2_layouts_free(&layouts);
}
