/* I publish complete ordinary declarations, independently of instruction flow. */
#include "../nanoisa/ownership_contracts.h"
#include "../nanoisa/retained_layouts.h"
static void ordinary_word(uint8_t *out,uint32_t value,unsigned width) {
    for(unsigned i=0;i<width;i++)out[i]=(uint8_t)(value>>(i*8));
}
static void publish_ordinary_authority(CG *cg,ASTNode *program) {
    NvmModule *module=cg->module;
    if(!cg->struct_count || cg->enum_count || cg->union_count || cg->extern_count ||
       module->layout_size || module->ownership_size)return;
    ASTNode *records[MAX_STRUCT_DEFS]={0};unsigned count=0;
    for(int i=0;i<program->as.program.count;i++) {
        ASTNode *node=bytecode_declaration(program->as.program.items[i]);
        if(node->type==AST_IMPORT || node->type==AST_ENUM_DEF || node->type==AST_UNION_DEF)return;
        if(node->type!=AST_STRUCT_DEF)continue;
        if(count>=(unsigned)cg->struct_count || node->as.struct_def.is_resource ||
           node->as.struct_def.is_extern || strchr(node->as.struct_def.name,'<') ||
           strcmp(node->as.struct_def.name,cg->structs[count].name))return;
        records[count++]=node;
    }
    if(count!=(unsigned)cg->struct_count)return;
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
    for(unsigned i=0;i<count && supported;i++) {
        ASTNode *node=records[i];int fields=node->as.struct_def.field_count;
        if(fields<0 || fields>UINT16_MAX){supported=false;break;}
        items[i]=(NvmV2Layout){NVM_V2_LAYOUT_STRUCT,(uint16_t)fields,NVM_V2_NO_INDEX,NULL};
        items[i].fields=fields?calloc((size_t)fields,sizeof *items[i].fields):NULL;
        if(fields && !items[i].fields){cg_error(cg,0,"I cannot allocate ordinary record fields");supported=false;break;}
        for(int j=0;j<fields;j++) {
            TypeInfo *info=node->as.struct_def.field_type_info?node->as.struct_def.field_type_info[j]:NULL;
            if(!info || info->type_param_count || info->element_type){supported=false;break;}
            uint8_t tag=ordinary_slot_tag(info->base_type);uint32_t child=NVM_V2_NO_INDEX;
            if(tag==TAG_STRUCT) {
                const char *name=node->as.struct_def.field_type_names?node->as.struct_def.field_type_names[j]:NULL;
                if(name)for(unsigned k=0;k<i;k++)if(!strcmp(name,records[k]->as.struct_def.name))child=k;
                if(child==NVM_V2_NO_INDEX){supported=false;break;}
            }
            if(tag==TAG_COUNT){supported=false;break;}
            items[i].fields[j]=(NvmV2LayoutField){tag,child,NVM_V2_NO_INDEX};
        }
    }
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
            if(conflict || tag==TAG_COUNT){supported=false;break;}
            data[at]=tag;ordinary_word(data+at+4,NVM_V2_NO_INDEX,4);at+=8;
        }
    }
    if(supported) {
        /* My shallow staged view owns only the two new optional buffers. */
        NvmModule staged=*module;staged.layout_data=NULL;staged.layout_size=0;
        staged.ownership_data=data;staged.ownership_size=(uint32_t)size;
        NvmV2Result result=nvm_retain_layouts(&staged,&layouts);bool needs=false;
        if(result==NVM_V2_OK)result=nvm_ownership_contracts_validate(&staged,&needs);
        if(result==NVM_V2_OK && !needs) {
            module->layout_data=staged.layout_data;module->layout_size=staged.layout_size;
            module->ownership_data=data;module->ownership_size=(uint32_t)size;data=NULL;
        } else {free(staged.layout_data);cg_error(cg,0,"I cannot validate ordinary declaration publication");}
    }
    free(data);nvm_v2_layouts_free(&layouts);
}
