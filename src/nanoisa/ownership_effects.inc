/* I retain operation signatures before granting any affine effect execution.
 * EFFECTS revision 1: u32 count, then rows ordered by key string index:
 * u32 key, u16 parameter count, u16 reserved, result descriptor, parameters.
 * Each descriptor is u8 tag, u8 mode (zero), u16 reserved, u32 layout.
 * Revision 2 prefixes a count and backward-linked element descriptors. ARRAY
 * layout indexes that table; nominal descriptors still index retained layouts.
 * Revision 3 inserts a u32 constructor count and (function, byte offset,
 * element index) u32 triples after the element table.
 * I retain declarations here; borrowed continuations need a separate contract. */
static NvmV2Result effect_value_read(NvmV2Cursor *cursor,
        const NvmV2Layouts *layouts,const uint8_t *flags,bool result_value,
        uint32_t earlier_types,bool arrays,
        NvmOwnershipEffectValue *out) {
    uint8_t mode;uint16_t reserved;NvmV2Result result;
    NvmOwnershipEffectValue value={0};
    if ((result=nvm_v2_u8(cursor,&value.tag))!=NVM_V2_OK ||
        (result=nvm_v2_u8(cursor,&mode))!=NVM_V2_OK ||
        (result=nvm_v2_u16(cursor,&reserved))!=NVM_V2_OK ||
        (result=nvm_v2_u32(cursor,&value.layout))!=NVM_V2_OK) return result;
    if (mode || reserved) return NVM_V2_ERR_RESERVED_FLAGS;
    if (scalar(value.tag) || value.tag==TAG_STRING ||
        (result_value && value.tag==TAG_VOID)) {
        if (value.layout!=NVM_V2_NO_INDEX) return NVM_V2_ERR_SECTION_TYPE;
    } else if (value.tag==TAG_STRUCT || value.tag==TAG_UNION) {
        if (value.layout>=layouts->count) return NVM_V2_ERR_INDEX_RANGE;
        uint16_t kind=value.tag==TAG_STRUCT?NVM_V2_LAYOUT_STRUCT:NVM_V2_LAYOUT_UNION;
        if (layouts->items[value.layout].kind!=kind ||
            !(flags[value.layout]&NVM_LAYOUT_COMPLETE)) return NVM_V2_ERR_SECTION_TYPE;
    } else if (arrays && value.tag==TAG_ARRAY) {
        if (value.layout>=earlier_types) return NVM_V2_ERR_INDEX_RANGE;
    } else return NVM_V2_ERR_SECTION_TYPE;
    *out=value;return NVM_V2_OK;
}

typedef struct { uint32_t function,pc,element; } EffectConstructorQuery;
/* I bind empty constructors to exact element identities, never inferred tags.
 * Rows are ordered by function and relative byte offset; I decode the prefix
 * once per function to reject offsets inside operands and malformed prefixes. */
static NvmV2Result effect_constructors_read(NvmV2Cursor *cursor,const NvmModule *module,
        const NvmOwnershipEffectTypes *types,EffectConstructorQuery *query) {
    uint32_t count,prior_function=UINT32_MAX,prior_site=0,pc=0;
    NvmV2Result result;
    if ((result=nvm_v2_u32(cursor,&count))!=NVM_V2_OK) return result;
    if (count>NVM_OWNERSHIP_MAX_ARRAY_CONSTRUCTORS) return NVM_V2_ERR_INDEX_RANGE;
    for (uint32_t i=0;i<count;i++) {
        uint32_t function,site,element;
        if ((result=nvm_v2_u32(cursor,&function))!=NVM_V2_OK ||
            (result=nvm_v2_u32(cursor,&site))!=NVM_V2_OK ||
            (result=nvm_v2_u32(cursor,&element))!=NVM_V2_OK) return result;
        if (!module->functions || function>=module->function_count || element>=types->count)
            return NVM_V2_ERR_INDEX_RANGE;
        if (i && (function<prior_function || (function==prior_function && site<=prior_site)))
            return NVM_V2_ERR_SECTION_TYPE;
        const NvmFunctionEntry *fn=&module->functions[function];
        if (!module->code || fn->code_offset>module->code_size ||
            fn->code_length>module->code_size-fn->code_offset || site>=fn->code_length)
            return NVM_V2_ERR_INDEX_RANGE;
        if (function!=prior_function) pc=0;
        DecodedInstruction instruction;uint32_t width;
        while (pc<site) {
            width=isa_decode(module->code+fn->code_offset+pc,fn->code_length-pc,&instruction);
            if (!width) return NVM_V2_ERR_SECTION_TYPE;
            pc+=width;
        }
        if (pc!=site) return NVM_V2_ERR_SECTION_TYPE;
        width=isa_decode(module->code+fn->code_offset+pc,fn->code_length-pc,&instruction);
        if (!width || (instruction.opcode!=OP_ARR_NEW &&
                !(instruction.opcode==OP_ARR_LITERAL && !instruction.operands[1].u16)) ||
            instruction.operands[0].u8!=types->items[element].tag)
            return NVM_V2_ERR_SECTION_TYPE;
        if (query && query->function==function && query->pc==site) query->element=element;
        prior_function=function;prior_site=site;pc+=width;
    }
    return NVM_V2_OK;
}

static NvmV2Result effects_read(NvmOwnershipExtensionView view,
        const NvmModule *module,const NvmV2Layouts *layouts,const uint8_t *flags,
        uint32_t wanted,NvmOwnershipEffect *out,NvmOwnershipEffectTypes *types_out,
        EffectConstructorQuery *constructor) {
    NvmV2Cursor cursor;nvm_v2_cursor_init(&cursor,view.data,view.size);
    uint32_t count,keys[NVM_OWNERSHIP_MAX_EFFECTS];NvmV2Result result;
    NvmOwnershipEffect selected={0};bool found=false;
    NvmOwnershipEffectTypes types={0};
    if (view.revision>=NVM_OWNERSHIP_EFFECT_ARRAY_REVISION) {
        if ((result=nvm_v2_u32(&cursor,&types.count))!=NVM_V2_OK) return result;
        if (!types.count || types.count>NVM_OWNERSHIP_MAX_EFFECT_TYPES)
            return NVM_V2_ERR_INDEX_RANGE;
        for (uint32_t i=0;i<types.count;i++)
            if ((result=effect_value_read(&cursor,layouts,flags,false,i,true,&types.items[i]))!=NVM_V2_OK)
                return result;
    }
    EffectConstructorQuery selected_constructor=constructor?*constructor:(EffectConstructorQuery){0};
    selected_constructor.element=NVM_V2_NO_INDEX;
    if (view.revision==NVM_OWNERSHIP_EFFECT_CONSTRUCTOR_REVISION &&
        (result=effect_constructors_read(&cursor,module,&types,&selected_constructor))!=NVM_V2_OK)
        return result;
    const char *requested=wanted==NVM_V2_NO_INDEX?NULL:nvm_get_string(module,wanted);
    if (wanted!=NVM_V2_NO_INDEX && !requested) return NVM_V2_ERR_INDEX_RANGE;
    if ((result=nvm_v2_u32(&cursor,&count))!=NVM_V2_OK) return result;
    if (!count || count>NVM_OWNERSHIP_MAX_EFFECTS) return NVM_V2_ERR_INDEX_RANGE;
    for (uint32_t i=0;i<count;i++) {
        NvmOwnershipEffect row={0};uint16_t reserved;
        if ((result=nvm_v2_u32(&cursor,&row.key_idx))!=NVM_V2_OK ||
            (result=nvm_v2_u16(&cursor,&row.parameter_count))!=NVM_V2_OK ||
            (result=nvm_v2_u16(&cursor,&reserved))!=NVM_V2_OK) return result;
        if (reserved) return NVM_V2_ERR_RESERVED_FLAGS;
        if (row.parameter_count>NVM_OWNERSHIP_MAX_EFFECT_PARAMETERS ||
            row.key_idx>=module->string_count) return NVM_V2_ERR_INDEX_RANGE;
        const char *key=nvm_get_string(module,row.key_idx);
        uint32_t length=nvm_get_string_len(module,row.key_idx);
        if (!key || !length || memchr(key,0,length) || (i && row.key_idx<=keys[i-1]))
            return NVM_V2_ERR_SECTION_TYPE;
        for (uint32_t j=0;j<i;j++)
            if (!strcmp(key,nvm_get_string(module,keys[j]))) return NVM_V2_ERR_SECTION_TYPE;
        keys[i]=row.key_idx;
        if ((result=effect_value_read(&cursor,layouts,flags,true,types.count,view.revision>=NVM_OWNERSHIP_EFFECT_ARRAY_REVISION,&row.result))!=NVM_V2_OK)
            return result;
        for (uint16_t p=0;p<row.parameter_count;p++)
            if ((result=effect_value_read(&cursor,layouts,flags,false,types.count,view.revision>=NVM_OWNERSHIP_EFFECT_ARRAY_REVISION,&row.parameters[p]))!=NVM_V2_OK)
                return result;
        if (requested && nvm_get_string_len(module,wanted)==length &&
            !memcmp(requested,key,length)) { selected=row;found=true; }
    }
    if (cursor.pos!=cursor.size) return NVM_V2_ERR_SECTION_RANGE;
    if (wanted!=NVM_V2_NO_INDEX && !found) return NVM_V2_ERR_INDEX_RANGE;
    if (out && found) *out=selected;
    if (types_out) *types_out=types;
    if (constructor) *constructor=selected_constructor;
    return NVM_V2_OK;
}
