/* I inspect declarations only. No bytecode or host callback executes. */
#include "portable_host_plan.h"
#include "isa.h"
#include "verifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"I failed %s:%d: %s\n",__FILE__,__LINE__,#x); abort(); } } while (0)
enum { QUERY=1, DECODE=2, STACK=4, TYPES=8 };
typedef struct { void *pointer; size_t bytes; } Allocation;
static Allocation live[256];
static size_t live_count, live_bytes, requested_bytes, requests;
static unsigned failed_domains;
static long fail_at=-1;
static bool persistent;
static unsigned domains[256];
static bool fail(unsigned domain, size_t bytes) {
    CHECK(requests < sizeof domains / sizeof domains[0]);
    domains[requests]=domain; requested_bytes+=bytes;
    bool failed=fail_at>=0 && (persistent ? requests>=(size_t)fail_at : requests==(size_t)fail_at);
    requests++;
    if (failed) failed_domains|=domain;
    return failed;
}
static void remember(void *p,size_t n) {
    if (!p) return;
    CHECK(live_count<sizeof live/sizeof live[0]);
    live[live_count++]=(Allocation){p,n};live_bytes+=n;
}
static void forget(void *p) {
    if (!p) return;
    for(size_t i=0;i<live_count;i++) if(live[i].pointer==p) {
        live_bytes-=live[i].bytes;live[i]=live[--live_count];return;
    }
    /* An untracked instrumented release is a fixture error. */
    CHECK(false);
}
static void *allocate(unsigned domain,size_t n,bool zero) {
    if(fail(domain,n))return NULL;
    void *p=zero?calloc(1,n):malloc(n);remember(p,n);return p;
}
static void *resize(unsigned domain,void *p,size_t n) {
    CHECK(n>0);
    if(fail(domain,n))return NULL;
    size_t index=live_count;
    if(p) { for(size_t i=0;i<live_count;i++)if(live[i].pointer==p){index=i;break;} CHECK(index<live_count); }
    bool had_pointer=p!=NULL;
    void *next=realloc(p,n);
    if(!next)return NULL;
    if(had_pointer){live_bytes-=live[index].bytes;live[index]=(Allocation){next,n};live_bytes+=n;}
    else remember(next,n);
    return next;
}
#define HOOKS(name,domain) \
void *name##_malloc(size_t n){return allocate(domain,n,false);} \
void *name##_calloc(size_t n,size_t w){CHECK(!w || n<=SIZE_MAX/w);return allocate(domain,n*w,true);} \
void *name##_realloc(void *p,size_t n){return resize(domain,p,n);} \
void name##_free(void *p){forget(p);free(p);}
HOOKS(read_query,QUERY)
HOOKS(read_decode,DECODE)
HOOKS(read_stack,STACK)
HOOKS(read_types,TYPES)
static void reset(long failure,bool prefix) {
    CHECK(!live_count && !live_bytes);requests=requested_bytes=0;failed_domains=0;
    fail_at=failure;persistent=prefix;
}
static void append(NvmModule *m,const uint8_t *code,uint32_t size) {
    uint32_t old=m->code_size;CHECK(nvm_append_code(m,code,size)==old);
    CHECK(m->code_size==old+size && !memcmp(m->code+old,code,size));
}
static uint32_t string(NvmModule *m,const char *s) {
    uint32_t index=nvm_add_string(m,s,(uint32_t)strlen(s));CHECK(index!=UINT32_MAX);return index;
}
static NvmModule *module(void) {
    NvmModule *m=nvm_module_new();CHECK(m);
    uint32_t empty=string(m,""),main_name=string(m,"main"),helper_name=string(m,"unused_identity");
    uint32_t path=string(m,"no-file-is-opened"),name;
    const char *aliases[]={"file_read","vm_file_read","nl_os_file_read"};
    uint8_t parameter=TAG_STRING;
    for(uint32_t i=0;i<3;i++) { name=string(m,aliases[i]);CHECK(nvm_add_import(m,empty,name,1,TAG_STRING,&parameter)==i); }
    uint8_t main_code[]={OP_PUSH_STR,0,0,0,0,OP_CALL_EXTERN,0,0,0,0,OP_POP,OP_PUSH_I64,0,0,0,0,0,0,0,0,OP_RET};
    main_code[1]=(uint8_t)path;append(m,main_code,sizeof main_code);
    NvmFunctionEntry f={.name_idx=main_name,.code_length=sizeof main_code,.result_tag=TAG_INT,.result_count=1};
    CHECK(nvm_add_function(m,&f)==0);
    uint8_t helper[]={OP_LOAD_LOCAL,0,0,OP_RET};
    f=(NvmFunctionEntry){.name_idx=helper_name,.arity=1,.local_count=1,.code_offset=m->code_size,
        .code_length=sizeof helper,.result_tag=TAG_STRING,.result_count=1};
    append(m,helper,sizeof helper);CHECK(nvm_add_function(m,&f)==1);
    CHECK(nvm_set_function_param_types(m,1,&parameter,1));
    m->header.flags|=NVM_FLAG_HAS_MAIN;m->header.entry_point=0;return m;
}
static NvmPortableReadPlan *expect(NvmModule *m,NvmPortableReadStatus wanted) {
    unsigned char sentinel;NvmPortableReadPlan *p=(void *)&sentinel;
    NvmPortableReadResult r=nvm_portable_read_plan(m,&p);
    if(r.status!=wanted)fprintf(stderr,"I expected status%d, got%d f%u pc%u import%u: %s\n",wanted,r.status,r.function_index,r.pc,r.import_index,r.message);
    CHECK(r.status==wanted && r.message);
    if(wanted==NVM_PORTABLE_READ_PREPARED){CHECK(p && p!=(void *)&sentinel);return p;}
    CHECK(p==(void *)&sentinel);return NULL;
}
static void positive(void) {
    puts("phase: copied declarations, input destruction and closed profiles");
    NvmModule *m=module();reset(-1,false);
    NvmPortableReadPlan *p=expect(m,NVM_PORTABLE_READ_PREPARED);
    NvmPortableReadCounts c;CHECK(nvm_portable_read_plan_counts(p,&c));
    CHECK(c.entry_function==0 && c.function_count==2 && c.import_count==3 && c.instruction_count==7);
    CHECK(c.module_bytes>m->code_size && c.module_bytes<=NVM_PORTABLE_READ_MAX_BYTES);
    CHECK(requested_bytes<=c.allocation_bound && c.allocation_bound<=NVM_PORTABLE_READ_MAX_BYTES);
    CHECK(nvm_verify(m).ok);
    for(int profile=NVM_PROFILE_CLOSED_SCALAR;profile<=NVM_PROFILE_CLOSED_MANAGED_STRINGS;profile++) {
        NvmVerifyResult r=nvm_verify_profile(m,(NvmVerifyProfile)profile);
        CHECK(!r.ok && strstr(r.error_msg,"without imports"));
    }
    uint32_t indices[3];for(unsigned i=0;i<3;i++)indices[i]=m->imports[i].function_name_idx;
    nvm_module_free(m);
    const char *aliases[]={"file_read","vm_file_read","nl_os_file_read"};
    for(uint32_t i=0;i<3;i++) {
        NvmPortableReadImport row;CHECK(nvm_portable_read_plan_import(p,i,&row));
        CHECK(row.import_index==i && row.namespace_string_index==0 && row.symbol_string_index==indices[i]);
        CHECK(row.operation==NVM_PORTABLE_HOST_READ_TEXT && row.revision==1);
        CHECK(row.argument_ownership==NVM_PORTABLE_HOST_BORROW_ROOTED_ARGUMENT && row.result_ownership==NVM_PORTABLE_HOST_COPY_MANAGED_RESULT);
        CHECK(row.parameter_count==1 && row.parameter_tag==TAG_STRING && row.result_count==1 && row.result_tag==TAG_STRING);
        CHECK(row.import_kind==NVM_IMPORT_FFI && !row.namespace_length && !row.namespace_bytes[0]);
        CHECK(row.symbol_length==strlen(aliases[i]) && !strcmp(row.symbol_bytes,aliases[i]));
        NvmPortableReadImport saved=row;
        CHECK(!nvm_portable_read_plan_import(p,3,&row) && !memcmp(&row,&saved,sizeof row));
        CHECK(!nvm_portable_read_plan_import(NULL,0,&row) && !memcmp(&row,&saved,sizeof row));
    }
    NvmPortableReadCounts saved=c;CHECK(!nvm_portable_read_plan_counts(NULL,&c) && !memcmp(&c,&saved,sizeof c));
    CHECK(!nvm_portable_read_plan_counts(p,NULL) && !nvm_portable_read_plan_import(p,0,NULL));
    nvm_portable_read_plan_free(p);nvm_portable_read_plan_free(NULL);CHECK(!live_count && !live_bytes);
}
static void declarations(void) {
    puts("phase: exact declarations and complete unused-function checks");
    NvmModule *m=module();reset(-1,false);
#define MUTATE(field,value,status) do { uint32_t saved_value=(field);(field)=(value);expect(m,status);(field)=saved_value; } while(0)
    MUTATE(m->header.flags,0,NVM_PORTABLE_READ_INVALID);
    MUTATE(m->header.entry_point,2,NVM_PORTABLE_READ_INVALID);
    MUTATE(m->header.magic[0],0,NVM_PORTABLE_READ_INVALID);
    MUTATE(m->imports[1].kind,NVM_IMPORT_COPROCESS,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->imports[1].kind,NVM_IMPORT_ARTIFACT,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->imports[1].kind,255,NVM_PORTABLE_READ_INVALID);
    MUTATE(m->imports[1].module_name_idx,1,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->imports[1].param_count,0,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->imports[1].return_type,TAG_INT,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->import_param_types[1][0],TAG_VOID,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->import_param_types[1][0],TAG_COUNT,NVM_PORTABLE_READ_INVALID);
    MUTATE(m->imports[1].function_name_idx,m->string_count,NVM_PORTABLE_READ_INVALID);
    uint32_t nul_name=nvm_add_string(m,"bad\0ns",6);CHECK(nul_name!=UINT32_MAX);
    MUTATE(m->imports[1].module_name_idx,nul_name,NVM_PORTABLE_READ_INVALID);
    uint8_t *tags=m->import_param_types[1];m->import_param_types[1]=NULL;expect(m,NVM_PORTABLE_READ_INVALID);m->import_param_types[1]=tags;
    tags=m->function_param_types[1];m->function_param_types[1]=NULL;expect(m,NVM_PORTABLE_READ_INVALID);m->function_param_types[1]=tags;
    MUTATE(m->function_param_types[1][0],TAG_VOID,NVM_PORTABLE_READ_UNSUPPORTED);
    uint32_t unknown=string(m,"file_read_extra"),old=m->imports[2].function_name_idx;
    m->imports[2].function_name_idx=unknown;expect(m,NVM_PORTABLE_READ_UNSUPPORTED);m->imports[2].function_name_idx=old;
    char *symbol=m->strings[old];char saved=symbol[3];symbol[3]=0;expect(m,NVM_PORTABLE_READ_INVALID);symbol[3]=saved;
    MUTATE(m->functions[1].code_offset,m->code_size+1,NVM_PORTABLE_READ_INVALID);
    MUTATE(m->functions[1].code_length,UINT32_MAX,NVM_PORTABLE_READ_INVALID);
    MUTATE(m->functions[1].code_offset,0,NVM_PORTABLE_READ_INVALID);
    MUTATE(m->code[m->functions[1].code_offset+1],255,NVM_PORTABLE_READ_INVALID);
    MUTATE(m->code[6],255,NVM_PORTABLE_READ_INVALID); /* Invalid CALL_EXTERN operand. */
    MUTATE(m->code[m->functions[1].code_offset],OP_PUSH_I64,NVM_PORTABLE_READ_INVALID); /* Truncated unused body. */
    MUTATE(m->functions[1].upvalue_count,1,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->layout_size,1,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->ownership_size,1,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->passive_size,1,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->module_ref_count,1,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->callback_contract_count,1,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->service_size,1,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->imports[0].kind,NVM_IMPORT_SERVICE,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->code[m->functions[1].code_offset],OP_FILE_DROP_STACK,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->code[0],OP_FILE_DROP_STACK,NVM_PORTABLE_READ_UNSUPPORTED);
    MUTATE(m->code[0],OP_REGION_BEGIN,NVM_PORTABLE_READ_UNSUPPORTED);
    unsigned char sentinel;NvmPortableReadPlan *out=(void *)&sentinel;
    uint8_t old_tag=m->import_param_types[1][0];m->import_param_types[1][0]=TAG_INT;
    NvmPortableReadResult located=nvm_portable_read_plan(m,&out);
    CHECK(located.status==NVM_PORTABLE_READ_UNSUPPORTED && located.import_index==1 &&
          located.function_index==NVM_PORTABLE_READ_NO_INDEX && located.pc==NVM_PORTABLE_READ_NO_INDEX && out==(void *)&sentinel);
    m->import_param_types[1][0]=old_tag;
    uint32_t pc=m->functions[1].code_offset;uint8_t old_op=m->code[pc];m->code[pc]=OP_PUSH_I64;
    located=nvm_portable_read_plan(m,&out);
    CHECK(located.status==NVM_PORTABLE_READ_INVALID && located.function_index==1 && located.pc==pc && out==(void *)&sentinel);
    m->code[pc]=old_op;
    CHECK(nvm_portable_read_plan(m,NULL).status==NVM_PORTABLE_READ_INVALID);
    expect(NULL,NVM_PORTABLE_READ_INVALID);
    nvm_portable_read_plan_free(expect(m,NVM_PORTABLE_READ_PREPARED));
    nvm_module_free(m);CHECK(!live_count);
#undef MUTATE
}
static NvmModule *large(uint32_t nops) {
    NvmModule *m=module();
    uint8_t *code=calloc((size_t)nops+10,1);CHECK(code);code[nops]=OP_PUSH_I64;code[nops+9]=OP_RET;
    free(m->code);m->code=code;m->code_size=m->code_capacity=nops+10;
    m->functions[0].code_offset=0;m->functions[0].code_length=m->code_size;
    free(m->function_param_types[1]);m->function_param_types[1]=NULL;m->function_count=1;
    return m;
}
static void limits(void) {
    puts("phase: independent preallocation limits");
    NvmModule *m=module();
#define LIMIT(field,value) do { uint32_t saved_value=(field);(field)=(value);reset(-1,false);expect(m,NVM_PORTABLE_READ_LIMIT);CHECK(!requests);(field)=saved_value; } while(0)
    LIMIT(m->function_count,NVM_PORTABLE_READ_MAX_FUNCTIONS+1);
    LIMIT(m->import_count,NVM_PORTABLE_READ_MAX_IMPORTS+1);
    LIMIT(m->string_count,NVM_MAX_STRINGS+1);
    LIMIT(m->code_size,NVM_PORTABLE_READ_MAX_BYTES+1);
    LIMIT(m->string_lengths[0],NVM_PORTABLE_READ_MAX_BYTES);
    nvm_module_free(m);
    m=large(NVM_PORTABLE_READ_MAX_INSTRUCTIONS);reset(-1,false);
    expect(m,NVM_PORTABLE_READ_LIMIT);CHECK(!requests);nvm_module_free(m);
    m=large(60000);reset(-1,false);expect(m,NVM_PORTABLE_READ_LIMIT);CHECK(!requests);nvm_module_free(m);
#undef LIMIT
    m=large(0); /* No CALL_EXTERN remains; no imports is genuinely unselected. */
    uint32_t imports=m->import_count;m->import_count=0;reset(-1,false);
    expect(m,NVM_PORTABLE_READ_NOT_SELECTED);m->import_count=imports;nvm_module_free(m);CHECK(!live_count);
}
#ifndef READ_LINKED
static void faults(void) {
    puts("phase: exact allocation domains, persistent prefixes and transient failures");
    NvmModule *m=module();NvmModule saved=*m;
    uint8_t code[25],param=m->function_param_types[1][0];memcpy(code,m->code,sizeof code);
    NvmFunctionEntry functions[2];NvmImportEntry imports[3];
    memcpy(functions,m->functions,sizeof functions);memcpy(imports,m->imports,sizeof imports);
    char strings[7][32];uint32_t lengths[7];char *string_pointers[7];
    uint8_t *import_parameters[3];
    for(unsigned i=0;i<3;i++)import_parameters[i]=m->import_param_types[i];
    uint8_t *function_parameters=m->function_param_types[1];
    CHECK(m->string_count==7);
    for(unsigned i=0;i<7;i++){CHECK(m->string_lengths[i]<32);string_pointers[i]=m->strings[i];lengths[i]=m->string_lengths[i];memcpy(strings[i],m->strings[i],lengths[i]+1);}
    reset(-1,false);NvmPortableReadPlan *p=expect(m,NVM_PORTABLE_READ_PREPARED);
    size_t total=requests;CHECK(total<128 && total>10 && domains[total-1]==QUERY);
    nvm_portable_read_plan_free(p);
    unsigned invalid=0,memory=0,advisory=0;
    for(unsigned mode=0;mode<2;mode++)for(size_t n=0;n<total;n++) {
        reset((long)n,mode==0);unsigned char sentinel;p=(void *)&sentinel;
        NvmPortableReadResult r=nvm_portable_read_plan(m,&p);fail_at=-1;
        CHECK(failed_domains);
        if(r.status==NVM_PORTABLE_READ_INVALID) { CHECK(r.function_index==NVM_PORTABLE_READ_NO_INDEX && r.pc==NVM_PORTABLE_READ_NO_INDEX && r.import_index==NVM_PORTABLE_READ_NO_INDEX);CHECK((failed_domains&(DECODE|STACK)) && !(failed_domains&QUERY));CHECK(p==(void *)&sentinel);invalid++; }
        else if(r.status==NVM_PORTABLE_READ_MEMORY) { CHECK(failed_domains&QUERY);CHECK(p==(void *)&sentinel);memory++; }
        else { CHECK(r.status==NVM_PORTABLE_READ_PREPARED && failed_domains==TYPES);CHECK(p!=(void *)&sentinel);nvm_portable_read_plan_free(p);advisory++; }
        printf("fault mode=%s index=%zu domains=%u status=%d\n",mode?"transient":"prefix",n,failed_domains,r.status);
        CHECK(!live_count && !live_bytes && !memcmp(&saved,m,sizeof saved));
        CHECK(!memcmp(code,m->code,sizeof code) && !memcmp(functions,m->functions,sizeof functions) && !memcmp(imports,m->imports,sizeof imports));
        CHECK(m->function_param_types[1]==function_parameters && m->function_param_types[1][0]==param);
        for(unsigned i=0;i<3;i++)CHECK(m->import_param_types[i]==import_parameters[i] && m->import_param_types[i][0]==TAG_STRING);
        for(unsigned i=0;i<7;i++)CHECK(m->strings[i]==string_pointers[i] && m->string_lengths[i]==lengths[i] && !memcmp(strings[i],m->strings[i],lengths[i]+1));
        reset(-1,false);nvm_portable_read_plan_free(expect(m,NVM_PORTABLE_READ_PREPARED));CHECK(!live_count && !live_bytes);
    }
    CHECK(invalid && memory && advisory);
    printf("I retain %u structural INVALID, %u report MEMORY and %u advisory-only successful transient queries.\n",invalid,memory,advisory);
    nvm_module_free(m);
}
#endif
int main(void) {
    setbuf(stdout,NULL);positive();declarations();limits();
#ifndef READ_LINKED
    faults();
#else
    CHECK(!failed_domains);
#endif
    CHECK(!live_count && !live_bytes);
    printf("PASS: %u private portable read-text checks; no bytecode or host execution.\n",checks);
    return 0;
}
