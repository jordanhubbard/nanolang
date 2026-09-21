#include "nvm2llvm_record_array_private.h"
#ifdef NANO_RECORD_ARRAY_GENERATED_PRIVATE
#include "record_array_generated_private.h"
#include "record_array_structure_private.h"
#include "managed_runtime_ir.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "record_array_emission_private.inc"
#include "record_array_llvm_operations_private.inc"
static void rl_declarations(RgOutput *b,const char *returned,const char *parameter) {
    rg_write(b,"%%V = type { i64, i32 }\n%%Fn = type { i32, i32, i32, i32, i32, ptr, ptr }\n"
        "%%P = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr }\n"
        "%%View = type { ptr, i32 }\n%%Record = type { i32, i32 }\n%%Field = type { i32, i32, i32 }\n"
        "declare i32 @nrg_layout_field(i32)\ndeclare i32 @nrg_create(ptr, ptr)\n"
        "declare i32 @nrg_status(ptr)\ndeclare i32 @nrg_resume(ptr)\n"
        "declare void @nrg_fail(ptr, i32)\ndeclare void @nrg_call(ptr, i32, i32)\ndeclare void @nrg_return(ptr)\n"
        "declare i32 @nrg_order(ptr, ptr, ptr)\n");
    static const struct { const char *name,*arguments; } ordinary[]={
        {"peek", "ptr, i32, ptr"}, {"push_move", "ptr, ptr"}, {"replace", "ptr, i32, ptr"},
        {"record_new", "ptr, i32"}, {"truth", "ptr, ptr"}, {"equal", "ptr, ptr, ptr"},
        {"format", "ptr, i32"}, {"predicate", "ptr, i32"}};
    for(size_t i=0;i<sizeof ordinary/sizeof ordinary[0];i++)
        rg_write(b,"declare %si1 @nrg_%s(%s)\n",returned,ordinary[i].name,ordinary[i].arguments);
    rg_write(b,"declare %si1 @nrg_load(ptr, i1 %s, i32)\n",returned,parameter);
    rg_write(b,"declare %si1 @nrg_store(ptr, i1 %s, i32)\n",returned,parameter);
    rg_write(b,"declare %si1 @nrg_record_get(ptr, i32, i1 %s)\n",returned,parameter);
    rg_write(b,"declare %si1 @nrg_record_set(ptr, i32, i1 %s)\n",returned,parameter);
    rg_write(b,"declare %si1 @nrg_array_new(ptr, i32, i32, i1 %s)\n",returned,parameter);
    rg_write(b,"declare %si1 @nrg_case(ptr, i1 %s)\n",returned,parameter);
    rg_write(b,"declare %si1 @nrg_length(ptr, i1 %s)\n",returned,parameter);
    static const char *const names[]={"drop","dup","swap","array_get","array_set","array_push","array_pop","array_slice",
        "cast_int","cast_float","cast_string","concat","substring","trim","split","string_replace","character"};
    for(size_t i=0;i<sizeof names/sizeof names[0];i++)rg_write(b,"declare %si1 @nrg_%s(ptr)\n",returned,names[i]);
}
static void rl_check(RlBlock *s,const char *type,const char *pointer,const char *expected) {
    uint32_t value=rl_next(s),ok=rl_next(s);
    rg_write(s->out," %%v%u = load %s, ptr %s\n %%v%u = icmp eq %s %%v%u, %s\n",value,type,pointer,ok,type,value,expected);
    rl_guard(s,ok,"invalid");
}
static void rl_layout(RlBlock *s,uint32_t index,const char *expected) {
    uint32_t value=rl_next(s),ok=rl_next(s);
    rg_write(s->out," %%v%u = call i32 @nrg_layout_field(i32 %u)\n %%v%u = icmp eq i32 %%v%u, %s\n",value,index,ok,value,expected);
    rl_guard(s,ok,"invalid");
}
static void rl_abi(RlBlock *s) {
    char expression[256];uint32_t index=0;
    static const uint32_t constants[]={NRG_ABI,NRG_FRAMES,NRG_ROOTS,4,1};
    for(size_t i=0;i<sizeof constants/sizeof constants[0];i++) { rl_text(s->out,expression,sizeof expression,"%u",constants[i]);rl_layout(s,index++,expression); }
    static const char *const types[]={"%V","%Fn","%P","%View","%Record","%Field"};
    static const uint32_t fields[]={2,7,17,2,2,3};
    for(size_t i=0;i<sizeof fields/sizeof fields[0];i++) {
        rl_text(s->out,expression,sizeof expression,"ptrtoint (ptr getelementptr (%s, ptr null, i32 1) to i32)",types[i]);rl_layout(s,index++,expression);
        rl_text(s->out,expression,sizeof expression,"ptrtoint (ptr getelementptr ({ i8, %s }, ptr null, i32 0, i32 1) to i32)",types[i]);rl_layout(s,index++,expression);
        for(uint32_t j=0;j<fields[i];j++) { rl_text(s->out,expression,sizeof expression,"ptrtoint (ptr getelementptr (%s, ptr null, i32 0, i32 %u) to i32)",types[i],j);rl_layout(s,index++,expression); }
    }
    if(index!=NRG_LAYOUT_COUNT) { s->out->status=NVM_ARRAY_INVALID;return; }
    rl_layout(s,index,"4294967295");
}
NvmArrayEligibilityResult nvm2llvm_record_array_private(const NvmModule *module,bool wasm32,
    char **out,size_t *length,NvmRecordArrayGeneratedCost *cost) {
    NvmArrayEligibilityResult result={.status=NVM_ARRAY_INVALID};
    snprintf(result.message,sizeof result.message,"I require a complete private generated program and disjoint outputs.");
    if(!module || !out || !length || !cost)return result;
    NvmRecordArrayExecutionPlan *plan=NULL;
    result=nvm_prepare_record_array_execution(module,&plan);
    if(result.status!=NVM_ARRAY_ELIGIBLE)return result;
    RgOutput b={.status=NVM_ARRAY_ELIGIBLE};
    NvmRecordArrayExecutionCounts counts;
    NvmDeclarationCounts declarations;
    NvmHeader header;
    NvmRecordArrayExecutionFunction functions[256];
    NvmRecordArrayExecutionDescriptor records[256];
    NvmDeclarationLayout layouts[256];
    uint32_t starts[256],record_starts[256],fields_count=0,record_fields=0;
    NrgField *fields=NULL;
    NvmRecordArrayExecutionInstruction *instructions=NULL;
    /* I additionally reserve 4096 bytes for all named LLVM helper scratch.
     * The 256/256/128/128-byte buffers, two blocks, table-value arrays and
     * scalar temporaries fit this conservative reservation; it is not a
     * claim about compiler spills, libc formatting or total host stack. */
    uint64_t fixed=sizeof b+sizeof counts+sizeof declarations+sizeof header+sizeof functions+sizeof records+
        sizeof layouts+sizeof starts+sizeof record_starts+4096;
    if(!rg_charge(&b,fixed,fixed) || !nvm_record_array_execution_counts(plan,&counts) ||
       !nvm_record_array_execution_declaration_counts(plan,&declarations) ||
       !nvm_record_array_execution_header(plan,&header) ||
       !counts.functions || counts.functions>256 || counts.entry>=counts.functions ||
       (counts.initializer!=UINT32_MAX && counts.initializer>=counts.functions) || counts.globals>256 ||
       counts.records>256 || declarations.layouts>256 || counts.instructions>65536)goto invalid;
    uint32_t supported=0;
    for(uint32_t op=0;op<256;op++) {
        NvmRecordArrayExecutionInstruction recipe={0};
        bool active=rg_recipe((uint8_t)op,&recipe)!=0;
        if(active!=nvm_record_array_opcode_supported((uint8_t)op))goto invalid;
        supported+=active;
    }
    if(supported!=93)goto invalid;
    for(uint32_t i=0;i<counts.functions;i++)if(!nvm_record_array_execution_function(plan,i,&functions[i]))goto invalid;
    for(uint32_t i=0;i<declarations.layouts;i++) {
        if(!nvm_record_array_execution_layout(plan,i,&layouts[i]) || layouts[i].fields>65536-fields_count)goto invalid;
        starts[i]=fields_count;fields_count+=layouts[i].fields;
    }
    if(!rg_charge(&b,(uint64_t)(fields_count?fields_count:1)*sizeof *fields,(uint64_t)(fields_count?fields_count:1)*sizeof *fields+
        (uint64_t)declarations.bindings*32))goto invalid;
    fields=calloc(fields_count?fields_count:1,sizeof *fields);
    if(!fields) { b.status=NVM_ARRAY_MEMORY;goto invalid; }
    for(uint32_t i=0;i<declarations.layouts;i++)for(uint16_t j=0;j<layouts[i].fields;j++) {
        NvmV2LayoutField f;if(!nvm_record_array_execution_field(plan,i,j,&f))goto invalid;
        fields[starts[i]+j]=(NrgField){f.type_tag,f.nested_idx,0};
    }
    for(uint32_t i=0;i<declarations.bindings;i++) {
        NvmOrdinaryArrayBinding binding;NvmOrdinaryArrayType element;
        if(!nvm_record_array_execution_binding(plan,i,&binding) || binding.layout>=declarations.layouts ||
           binding.field>=layouts[binding.layout].fields ||
           !nvm_record_array_execution_type(plan,binding.element_type,&element))goto invalid;
        fields[starts[binding.layout]+binding.field].element=element.tag;
    }
    for(uint32_t i=0;i<counts.records;i++) {
        if(!nvm_record_array_execution_descriptor(plan,i,&records[i]) || records[i].ordinal!=i ||
           records[i].layout>=declarations.layouts || records[i].fields!=layouts[records[i].layout].fields)goto invalid;
        record_starts[i]=record_fields;record_fields+=records[i].fields;
    }
    uint64_t instruction_bytes=(uint64_t)(counts.instructions?counts.instructions:1)*sizeof *instructions;
    if(!rg_charge(&b,instruction_bytes,instruction_bytes))goto invalid;
    instructions=calloc(counts.instructions?counts.instructions:1,sizeof *instructions);
    if(!instructions) { b.status=NVM_ARRAY_MEMORY;goto invalid; }
    uint32_t total=0;
    for(uint32_t fi=0;fi<counts.functions;fi++) {
        NvmRecordArrayExecutionFunction *fn=&functions[fi];uint32_t pc=0;
        if(fn->instruction_start!=total || fn->instruction_count>counts.instructions-total)goto invalid;
        for(uint32_t j=0;j<fn->instruction_count;j++) {
            NvmRecordArrayExecutionInstruction *r=&instructions[total+j];
            if(!nvm_record_array_execution_instruction(plan,total+j,r) || r->function!=fi || r->pc!=pc ||
               !rg_fact(&b,plan,functions,counts.functions,r))goto invalid;
            pc=r->next_pc;
        }
        if(pc!=fn->signature.code_length)goto invalid;
        total+=fn->instruction_count;
    }
    if(total!=counts.instructions)goto invalid;
    for(uint32_t i=0;i<counts.instructions;i++) {
        const NvmRecordArrayExecutionInstruction *r=&instructions[i];
        const NvmRecordArrayExecutionFunction *f=&functions[r->function];
        for(uint8_t edge=0;edge<r->successor_count;edge++) {
            uint32_t target=r->successors[edge];
            if(target==f->signature.code_length)continue;
            uint32_t lo=f->instruction_start,hi=lo+f->instruction_count;
            while(lo<hi) {
                if(!rg_charge(&b,0,1))goto invalid;
                uint32_t middle=lo+(hi-lo)/2;
                if(instructions[middle].pc<target)lo=middle+1;else hi=middle;
            }
            if(lo==f->instruction_start+f->instruction_count || instructions[lo].pc!=target)goto invalid;
        }
    }

    /* I use only the generated target prefix, never its legacy runtime body. */
    (void)&nms_runtime_ir_native;(void)&nms_runtime_ir_wasm32;
    rg_write(&b,"%s",wasm32?nms_runtime_target_wasm32:nms_runtime_target_native);
    const char *bool_return=wasm32?NMS_BOOL_RETURN_WASM32:NMS_BOOL_RETURN_NATIVE;
    const char *bool_parameter=wasm32?NMS_BOOL_PARAMETER_WASM32:NMS_BOOL_PARAMETER_NATIVE;
    rl_declarations(&b,bool_return,bool_parameter);
    for(uint32_t i=0;i<counts.strings;i++) {
        uint32_t size;if(!nvm_record_array_execution_size(plan,NVM_RA_SNAPSHOT_STRING,i,&size))goto invalid;
        rg_write(&b,"@literal_%u = private constant [%u x i8] c\"",i,size?size:1);
        for(uint32_t j=0;j<size;j++) {
            uint8_t byte;if(!rg_charge(&b,0,1) || !nvm_record_array_execution_bytes(plan,NVM_RA_SNAPSHOT_STRING,i,j,1,&byte))goto invalid;
            rg_write(&b,"\\%02X",byte);
        }
        if(!size)rg_write(&b,"\\00");
    rg_write(&b,"\"\n");
    }
    rg_write(&b,"@literals = private constant [%u x %%View] [",counts.strings?counts.strings:1);
    for(uint32_t i=0;i<counts.strings;i++) {
        uint32_t size;if(!nvm_record_array_execution_size(plan,NVM_RA_SNAPSHOT_STRING,i,&size))goto invalid;
        rg_write(&b,"%s%%View { ptr @literal_%u, i32 %u }",i?", ":"",i,size);
    }
    if(!counts.strings)rg_write(&b,"%%View zeroinitializer");
    rg_write(&b,"]\n");
    rg_write(&b,"@records = private constant [%u x %%Record] [",counts.records?counts.records:1);
    for(uint32_t i=0;i<counts.records;i++)rg_write(&b,"%s%%Record { i32 %u, i32 %u }",i?", ":"",records[i].layout,records[i].fields);
    if(!counts.records)rg_write(&b,"%%Record zeroinitializer");
    rg_write(&b,"]\n");
    rg_write(&b,"@starts = private constant [%u x i32] [",counts.records?counts.records:1);
    for(uint32_t i=0;i<counts.records;i++)rg_write(&b,"%si32 %u",i?", ":"",record_starts[i]);
    if(!counts.records)rg_write(&b,"i32 0");
    rg_write(&b,"]\n");
    rg_write(&b,"@fields = private constant [%u x %%Field] [",record_fields?record_fields:1);
    for(uint32_t i=0;i<counts.records;i++)for(uint32_t j=0;j<records[i].fields;j++) {
        uint32_t k=record_starts[i]+j;NrgField f=fields[starts[records[i].layout]+j];
        rg_write(&b,"%s%%Field { i32 %u, i32 %u, i32 %u }",k?", ":"",f.tag,f.nested_layout,f.element);
    }
    if(!record_fields)rg_write(&b,"%%Field zeroinitializer");
    rg_write(&b,"]\n");
    for(uint32_t i=0;i<counts.functions;i++)if(functions[i].parameter_tags_present) {
        uint32_t arity=functions[i].signature.arity;rg_write(&b,"@parameters_%u = private constant [%u x i8] [",i,arity?arity:1);
        for(uint32_t j=0;j<arity;j++) {
            uint8_t tag;if(!nvm_record_array_execution_parameter(plan,i,(uint16_t)j,&tag))goto invalid;
            rg_write(&b,"%si8 %u",j?", ":"",tag);
        }
        if(!arity)rg_write(&b,"i8 0");
    rg_write(&b,"]\n");
    }
    rg_write(&b,"@functions = private constant [%u x %%Fn] [",counts.functions);
    for(uint32_t i=0;i<counts.functions;i++) {
        NvmRecordArrayExecutionFunction *f=&functions[i];
        rg_write(&b,"%s%%Fn { i32 %u, i32 %u, i32 %u, i32 %u, i32 %u, ptr ",i?", ":"",f->signature.local_count,f->signature.arity,f->signature.result_count,f->signature.result_tag,f->maximum_stack);
        if(f->parameter_tags_present)rg_write(&b,"@parameters_%u",i);else rg_write(&b,"null");
        rg_write(&b,", ptr @body_%u }",i);
    }
    rg_write(&b,"]\n@program = private constant %%P { i32 %u, i32 16, i32 8, i32 %u, i32 %u, i32 %u, i32 %u, i32 %u, i32 %u, i32 %u, i32 %u, i32 %u, ptr @functions, ptr @literals, ptr @records, ptr @starts, ptr @fields }\n",
        NRG_ABI,NRG_FRAMES,counts.functions,counts.entry,counts.initializer,counts.globals,(unsigned)!!(header.flags&NVM_FLAG_HAS_MAIN),counts.strings,counts.records,record_fields);
    for(uint32_t fi=0;fi<counts.functions;fi++) {
        const NvmRecordArrayExecutionFunction *f=&functions[fi];RlBlock block={&b,0,bool_return,bool_parameter};
        rg_write(&b,"define private void @body_%u(ptr %%p) {\nentry:\n",fi);
        for(unsigned scratch_index=0;scratch_index<3;scratch_index++) {
            char scratch="abr"[scratch_index];
            rg_write(&b," %%%c = alloca %%V, align 8\n %%%c_payload = getelementptr %%V, ptr %%%c, i32 0, i32 0\n %%%c_tag = getelementptr %%V, ptr %%%c, i32 0, i32 1\n",scratch,scratch,scratch,scratch,scratch);
        }
        rg_write(&b," %%resume = call i32 @nrg_resume(ptr %%p)\n switch i32 %%resume, label %%state_error [\n");
        for(uint32_t j=0;j<f->instruction_count;j++)rg_write(&b," i32 %u, label %%b%u\n",instructions[f->instruction_start+j].pc,instructions[f->instruction_start+j].pc);
        rg_write(&b," i32 %u, label %%b%u\n ]\n",f->signature.code_length,f->signature.code_length);
        for(uint32_t j=0;j<f->instruction_count;j++) {
            if(!rg_charge(&b,0,512))goto invalid;
            rl_operation(&block,&instructions[f->instruction_start+j]);
        }
        rg_write(&b,"b%u:\n call void @nrg_return(ptr %%p)\n ret void\n"
            "type_error:\n call void @nrg_fail(ptr %%p, i32 %u)\n ret void\n"
            "assert_error:\n call void @nrg_fail(ptr %%p, i32 %u)\n ret void\n"
            "state_error:\n call void @nrg_fail(ptr %%p, i32 %u)\n ret void\nerror:\n ret void\n}\n",
            f->signature.code_length,NRG_TYPE,NRG_ASSERT,NRG_STATE);
    }
    rg_write(&b,"define i32 @nrg_generated_create(ptr %%out) {\nentry:\n");
    RlBlock startup={&b,0,bool_return,bool_parameter};rl_abi(&startup);
    char pointer[256],expected[128];
    const uint32_t program_values[]={NRG_ABI,16,8,NRG_FRAMES,counts.functions,counts.entry,counts.initializer,counts.globals,(unsigned)!!(header.flags&NVM_FLAG_HAS_MAIN),counts.strings,counts.records,record_fields};
    for(uint32_t i=0;i<12;i++) {
        rl_text(&b,pointer,sizeof pointer,"getelementptr (%%P, ptr @program, i32 0, i32 %u)",i);
        rl_text(&b,expected,sizeof expected,"%u",program_values[i]);rl_check(&startup,"i32",pointer,expected);
    }
    static const char *const program_pointers[]={"@functions","@literals","@records","@starts","@fields"};
    for(uint32_t i=0;i<5;i++) {
        rl_text(&b,pointer,sizeof pointer,"getelementptr (%%P, ptr @program, i32 0, i32 %u)",i+12);
        rl_check(&startup,"ptr",pointer,program_pointers[i]);
    }
    for(uint32_t i=0;i<counts.functions;i++) {
        const NvmRecordArrayExecutionFunction *f=&functions[i];
        const uint32_t values[]={f->signature.local_count,f->signature.arity,f->signature.result_count,f->signature.result_tag,f->maximum_stack};
        for(uint32_t j=0;j<5;j++) {
            rl_text(&b,pointer,sizeof pointer,"getelementptr ([%u x %%Fn], ptr @functions, i32 0, i32 %u, i32 %u)",counts.functions,i,j);
            rl_text(&b,expected,sizeof expected,"%u",values[j]);rl_check(&startup,"i32",pointer,expected);
        }
        rl_text(&b,pointer,sizeof pointer,"getelementptr ([%u x %%Fn], ptr @functions, i32 0, i32 %u, i32 5)",counts.functions,i);
        if(f->parameter_tags_present)rl_text(&b,expected,sizeof expected,"@parameters_%u",i);else rl_text(&b,expected,sizeof expected,"null");
        rl_check(&startup,"ptr",pointer,expected);
        rl_text(&b,pointer,sizeof pointer,"getelementptr ([%u x %%Fn], ptr @functions, i32 0, i32 %u, i32 6)",counts.functions,i);
        rl_text(&b,expected,sizeof expected,"@body_%u",i);rl_check(&startup,"ptr",pointer,expected);
        if(f->parameter_tags_present)for(uint32_t j=0;j<f->signature.arity;j++) {
            uint8_t tag;if(!nvm_record_array_execution_parameter(plan,i,(uint16_t)j,&tag))goto invalid;
            rl_text(&b,pointer,sizeof pointer,"getelementptr ([%u x i8], ptr @parameters_%u, i32 0, i32 %u)",f->signature.arity?f->signature.arity:1,i,j);
            rl_text(&b,expected,sizeof expected,"%u",tag);rl_check(&startup,"i8",pointer,expected);
        }
    }
    for(uint32_t i=0;i<counts.strings;i++) {
        uint32_t size;if(!nvm_record_array_execution_size(plan,NVM_RA_SNAPSHOT_STRING,i,&size))goto invalid;
        rl_text(&b,pointer,sizeof pointer,"getelementptr ([%u x %%View], ptr @literals, i32 0, i32 %u, i32 0)",counts.strings,i);
        rl_text(&b,expected,sizeof expected,"@literal_%u",i);rl_check(&startup,"ptr",pointer,expected);
        rl_text(&b,pointer,sizeof pointer,"getelementptr ([%u x %%View], ptr @literals, i32 0, i32 %u, i32 1)",counts.strings,i);
        rl_text(&b,expected,sizeof expected,"%u",size);rl_check(&startup,"i32",pointer,expected);
    }
    for(uint32_t i=0;i<counts.records;i++) {
        for(uint32_t j=0;j<2;j++) {
            rl_text(&b,pointer,sizeof pointer,"getelementptr ([%u x %%Record], ptr @records, i32 0, i32 %u, i32 %u)",counts.records,i,j);
            rl_text(&b,expected,sizeof expected,"%u",j?records[i].fields:records[i].layout);rl_check(&startup,"i32",pointer,expected);
        }
        rl_text(&b,pointer,sizeof pointer,"getelementptr ([%u x i32], ptr @starts, i32 0, i32 %u)",counts.records,i);
        rl_text(&b,expected,sizeof expected,"%u",record_starts[i]);rl_check(&startup,"i32",pointer,expected);
        for(uint32_t j=0;j<records[i].fields;j++) {
            NrgField f=fields[starts[records[i].layout]+j];uint32_t values[]={f.tag,f.nested_layout,f.element};
            for(uint32_t k=0;k<3;k++) {
                rl_text(&b,pointer,sizeof pointer,"getelementptr ([%u x %%Field], ptr @fields, i32 0, i32 %u, i32 %u)",record_fields,record_starts[i]+j,k);
                rl_text(&b,expected,sizeof expected,"%u",values[k]);rl_check(&startup,"i32",pointer,expected);
            }
        }
    }
    rg_write(&b," %%created = call i32 @nrg_create(ptr @program, ptr %%out)\n ret i32 %%created\ninvalid:\n ret i32 %u\n}\n",NRG_STATE);
    if(b.status!=NVM_ARRAY_ELIGIBLE)goto invalid;
    *cost=(NvmRecordArrayGeneratedCost){counts.peak_bytes_reserved,counts.work_reserved,b.peak,b.work};
    *out=b.text;*length=b.used;b.text=NULL;
    free(fields);free(instructions);nvm_record_array_execution_free(plan);return result;
invalid:
    if(b.status==NVM_ARRAY_ELIGIBLE)b.status=NVM_ARRAY_INVALID;
    result.status=b.status;snprintf(result.message,sizeof result.message,"I could not complete private generated correspondence and emission.");
    free(b.text);free(fields);free(instructions);nvm_record_array_execution_free(plan);return result;
}
#endif
