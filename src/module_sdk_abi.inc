/* I retain a canonical, owned C-storage schema, never execution authority.
 * This file is included by module_builder.c and its owning parser fixture. */
#define MODULE_SDK_ABI_MAX_BYTES ((size_t)1024 * 1024)
#define MODULE_SDK_ABI_MAX_STEPS 1048576u
#define MODULE_SDK_ABI_MAX_TYPES 4096u
#define MODULE_SDK_ABI_MAX_FUNCTIONS 65536u
typedef struct { size_t bytes; uint32_t steps; } ModuleSdkAbiLimit;
static bool module_abi_charge(ModuleSdkAbiLimit *limit,size_t bytes,uint32_t steps) {
    if(bytes>limit->bytes||steps>limit->steps)return false;
    limit->bytes-=bytes;limit->steps-=steps;return true;
}
static bool module_abi_identifier(const char *s,bool path) {
    if(!s||!s[0])return false;
    bool first=true;
    for(const unsigned char *p=(const unsigned char *)s;*p;p++) {
        if(path&&*p=='.') {if(first)return false;first=true;continue;}
        bool letter=(*p>='a'&&*p<='z')||(*p>='A'&&*p<='Z')||*p=='_';
        if(!letter&&(first||*p<'0'||*p>'9'))return false;
        first=false;
    }
    return !first;
}
static bool module_abi_text(ModuleSdkAbiLimit *limit,const cJSON *j,unsigned form) {
    if(!cJSON_IsString(j)||!j->valuestring)return false;
    size_t n=strnlen(j->valuestring,MODULE_SDK_ABI_MAX_BYTES+1);
    if(!n||n>(form==1?255u:4096u)||!module_abi_charge(limit,n*6+64,(uint32_t)n+1))return false;
    for(size_t i=0;i<n;i++)if((unsigned char)j->valuestring[i]<32||(unsigned char)j->valuestring[i]==127)return false;
    return !form||module_abi_identifier(j->valuestring,form==2);
}
/* I reject duplicate/unknown keys before any canonical copy. */
static bool module_abi_fields(ModuleSdkAbiLimit *limit,const cJSON *object,
        const char *const *names,size_t count,const cJSON **values) {
    if(!cJSON_IsObject(object)||!module_abi_charge(limit,64,1))return false;
    for(size_t i=0;i<count;i++)values[i]=NULL;
    for(const cJSON *j=object->child;j;j=j->next) {
        if(!j->string||strnlen(j->string,65)>64||!module_abi_charge(limit,strlen(j->string)*6+64,1))return false;
        size_t i=0;while(i<count&&strcmp(j->string,names[i]))i++;
        if(i==count||values[i])return false;
        values[i]=j;
    }
    return true;
}
static bool module_abi_index(ModuleSdkAbiLimit *limit,const cJSON *j,uint32_t count) {
    if(!cJSON_IsNumber(j)||!module_abi_charge(limit,64,1))return false;
    return j->valuedouble>=0&&j->valuedouble<count&&j->valuedouble==(double)(uint32_t)j->valuedouble;
}
static bool module_abi_distinct(ModuleSdkAbiLimit *limit,const char *left,const char *right) {
    size_t work=strlen(left)+strlen(right)+1;
    return work<=UINT32_MAX&&module_abi_charge(limit,0,(uint32_t)work)&&strcmp(left,right)!=0;
}
static bool module_abi_strings(ModuleSdkAbiLimit *limit,const cJSON *j,unsigned form) {
    if(!cJSON_IsArray(j)||!module_abi_charge(limit,64,1))return false;
    uint32_t count=0;
    for(const cJSON *v=j->child;v;v=v->next) {
        if(count++==65535||!module_abi_text(limit,v,form))return false;
        for(const cJSON *prior=j->child;prior!=v;prior=prior->next) {
            if(!module_abi_distinct(limit,prior->valuestring,v->valuestring))return false;
        }
    }
    return true;
}
static bool module_abi_type(ModuleSdkAbiLimit *limit,const cJSON *row,uint32_t types) {
    static const char *const keys[]={"semantic","c_type","kind","fields","tag","variants","abi","element"};
    const cJSON *v[8];
    if(!module_abi_fields(limit,row,keys,8,v)||!module_abi_text(limit,v[0],0)||
       !module_abi_text(limit,v[1],1)||!module_abi_text(limit,v[2],1))return false;
    const char *kind=v[2]->valuestring;
    if(!strcmp(kind,"record")||!strcmp(kind,"tuple"))
        return !v[4]&&!v[5]&&!v[6]&&!v[7]&&module_abi_strings(limit,v[3],2);
    if(!strcmp(kind,"union")) {
        if(v[3]||v[6]||v[7]||!module_abi_text(limit,v[4],2)||!cJSON_IsArray(v[5])||!module_abi_charge(limit,64,1))return false;
        uint32_t count=0;
        for(const cJSON *variant=v[5]->child;variant;variant=variant->next) {
            static const char *const names[]={"tag","fields"};const cJSON *parts[2];
            if(count++==65535||!module_abi_fields(limit,variant,names,2,parts)||
               !module_abi_text(limit,parts[0],1)||!module_abi_strings(limit,parts[1],2))return false;
            for(const cJSON *prior=v[5]->child;prior!=variant;prior=prior->next) {
                const cJSON *tag=cJSON_GetObjectItemCaseSensitive(prior,"tag");
                if(!module_abi_distinct(limit,tag->valuestring,parts[0]->valuestring))return false;
            }
        }
        return count!=0;
    }
    if(!strcmp(kind,"enum"))return !v[3]&&!v[4]&&!v[6]&&!v[7]&&
        module_abi_strings(limit,v[5],1)&&v[5]->child;
    if(!strcmp(kind,"array"))return !v[3]&&!v[4]&&!v[5]&&module_abi_text(limit,v[6],1)&&
        !strcmp(v[6]->valuestring,"dyn_array_v2")&&module_abi_index(limit,v[7],types);
    if(v[3]||v[4]||v[5]||v[6]||v[7])return false;
    return !strcmp(kind,"int")||!strcmp(kind,"float")||!strcmp(kind,"bool")||
        !strcmp(kind,"u8")||!strcmp(kind,"string")||!strcmp(kind,"opaque")||!strcmp(kind,"function");
}
static bool module_abi_function(ModuleSdkAbiLimit *limit,const cJSON *row,uint32_t types) {
    static const char *const keys[]={"name","symbol","parameters","result"};const cJSON *v[4];
    if(!module_abi_fields(limit,row,keys,4,v)||!module_abi_text(limit,v[0],1)||
       !module_abi_text(limit,v[1],1)||!cJSON_IsArray(v[2])||!module_abi_charge(limit,64,1))return false;
    uint32_t count=0;
    for(const cJSON *p=v[2]->child;p;p=p->next)if(count++==65535||!module_abi_index(limit,p,types))return false;
    return cJSON_IsNull(v[3])?module_abi_charge(limit,64,1):module_abi_index(limit,v[3],types);
}
static bool module_abi_validate(const cJSON *schema) {
    ModuleSdkAbiLimit limit={MODULE_SDK_ABI_MAX_BYTES,MODULE_SDK_ABI_MAX_STEPS};
    static const char *const keys[]={"version","target","types","functions"};const cJSON *v[4];
    if(!module_abi_fields(&limit,schema,keys,4,v)||!cJSON_IsNumber(v[0])||v[0]->valuedouble!=1||
       !module_abi_text(&limit,v[1],0)||!cJSON_IsArray(v[2])||!cJSON_IsArray(v[3])||
       !module_abi_charge(&limit,192,3))return false;
    uint32_t types=0,functions=0;
    for(const cJSON *j=v[2]->child;j;j=j->next)if(types++==MODULE_SDK_ABI_MAX_TYPES||!module_abi_charge(&limit,0,1))return false;
    for(const cJSON *j=v[2]->child;j;j=j->next) {
        if(!module_abi_type(&limit,j,types))return false;
        const cJSON *name=cJSON_GetObjectItemCaseSensitive(j,"semantic");
        for(const cJSON *p=v[2]->child;p!=j;p=p->next) {
            const cJSON *prior=cJSON_GetObjectItemCaseSensitive(p,"semantic");
            if(!module_abi_distinct(&limit,name->valuestring,prior->valuestring))return false;
        }
    }
    for(const cJSON *j=v[3]->child;j;j=j->next) {
        if(functions++==MODULE_SDK_ABI_MAX_FUNCTIONS||!module_abi_function(&limit,j,types))return false;
        const cJSON *name=cJSON_GetObjectItemCaseSensitive(j,"name");
        for(const cJSON *p=v[3]->child;p!=j;p=p->next) {
            const cJSON *prior=cJSON_GetObjectItemCaseSensitive(p,"name");
            if(!module_abi_distinct(&limit,name->valuestring,prior->valuestring))return false;
        }
    }
    return true;
}
/* Validated object arity is at most eight and nesting is bounded by the
 * explicit schema grammar. I sort object keys, preserving ordered arrays. */
static cJSON *module_abi_copy(const cJSON *value) {
    if(cJSON_IsObject(value)) {
        const cJSON *fields[8];size_t count=0;
        for(const cJSON *j=value->child;j;j=j->next) {
            if(count==8)return NULL;
            size_t at=count++;while(at&&strcmp(fields[at-1]->string,j->string)>0){fields[at]=fields[at-1];at--;}
            fields[at]=j;
        }
        cJSON *copy=cJSON_CreateObject();if(!copy)return NULL;
        for(size_t i=0;i<count;i++) {
            cJSON *child=module_abi_copy(fields[i]);
            if(!child||!cJSON_AddItemToObject(copy,fields[i]->string,child)){cJSON_Delete(child);cJSON_Delete(copy);return NULL;}
        }
        return copy;
    }
    if(cJSON_IsArray(value)) {
        cJSON *copy=cJSON_CreateArray();if(!copy)return NULL;
        for(const cJSON *j=value->child;j;j=j->next) {
            cJSON *child=module_abi_copy(j);
            if(!child||!cJSON_AddItemToArray(copy,child)){cJSON_Delete(child);cJSON_Delete(copy);return NULL;}
        }
        return copy;
    }
    return cJSON_Duplicate(value,false);
}
static bool module_parse_typed_abi(const cJSON *json,ModuleBuildMetadata *meta) {
    const cJSON *schema=NULL;
    if(!cJSON_IsObject(json)||!meta||meta->typed_abi)return false;
    for(const cJSON *j=json->child;j;j=j->next)if(j->string&&!strcmp(j->string,"typed_abi")) {
        if(schema)return false;
        schema=j;
    }
    if(!schema)return true;
    if(!module_abi_validate(schema))return false;
    cJSON *copy=module_abi_copy(schema);if(!copy)return false;
    char *canonical=cJSON_PrintUnformatted(copy);cJSON_Delete(copy);
    if(!canonical)return false;
    size_t n=strlen(canonical);
    if(n>MODULE_SDK_ABI_MAX_BYTES){cJSON_free(canonical);return false;}
    /* cJSON may use caller-supplied hooks. The metadata's final string follows
     * its ordinary malloc/free ownership contract instead. */
    char *owned=malloc(n+1);
    if(owned)memcpy(owned,canonical,n+1);
    cJSON_free(canonical);if(!owned)return false;
    meta->typed_abi=owned;return true;
}
