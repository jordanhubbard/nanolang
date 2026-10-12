/* I preserve exact element facts and ownership on every admitted array edge. */
static bool array_index(Value v) { return v.tag==TAG_INT && !v.owned && !v.observation; }
static bool array_push_type(Frame *f,NvmAffineType type) {
    return push(f,(Value){.tag=type.tag,.layout=type.layout,.root=UINT16_MAX,
        .owned=nvm_affine_type_is_owned(f->locals,type),.variant=NVM_AFFINE_UNKNOWN_VARIANT});
}
static const char *array_step(Frame *f,const DecodedInstruction *in) {
    uint8_t op=in->opcode;NvmAffineType type,element;
    if (!nvm_affine_has_array_types(f->locals)) return "I require retained element types for array flow";
    if (op==OP_ARR_NEW || op==OP_ARR_LITERAL) {
        uint16_t count=op==OP_ARR_LITERAL?in->operands[1].u16:0;
        uint8_t tag=in->operands[0].u8;
        if (count>f->count) return "I require all declared array literal elements";
        if (!count) {
            if (!nvm_affine_array_tag_type(f->locals,tag,&type))
                return "I require an unambiguous retained element type for empty array construction";
        } else {
            Value first=f->stack[f->count-count];element=(NvmAffineType){first.tag,first.layout};
            if (element.tag!=tag || !nvm_affine_array_type(f->locals,element,&type))
                return "I require an exact retained array literal element type";
            for (uint16_t i=0;i<count;i++)
                if (!effect_value_matches(f->locals,f->stack[f->count-count+i],
                        (NvmOwnershipEffectValue){element.tag,element.layout}))
                    return "I require exact array literal element ownership in source order";
        }
        f->count-=count;
        return array_push_type(f,type)?NULL:"I cannot retain an array construction result";
    }
    uint16_t count=(op==OP_ARR_SET || op==OP_ARR_SLICE)?3:
        (op==OP_ARR_PUSH || op==OP_ARR_GET || op==OP_ARR_REMOVE)?2:1;
    if (f->count<count) return "I require every array operation operand";
    Value array=f->stack[f->count-count];type=(NvmAffineType){array.tag,array.layout};
    if (!nvm_affine_array_element(f->locals,type,&element)) return "I require an exact array operand";
    if (op==OP_ARR_LEN) {
        if (array.owned || (array.observation &&
            !nvm_affine_owner_access(f->locals,array.root,NULL,0,false)))
            return "I require a live owner observation for resource-array length";
        f->count--;return array_push_type(f,(NvmAffineType){TAG_INT,NVM_V2_NO_INDEX})?NULL:
            "I cannot retain array length";
    }
    if (array.observation) return "I cannot mutate or extract through an array owner observation";
    if (op!=OP_ARR_PUSH && array.owned)
        return "I require explicit resource-element extraction and disposal contracts";
    if ((op==OP_ARR_GET || op==OP_ARR_REMOVE || op==OP_ARR_SET || op==OP_ARR_SLICE) &&
        !array_index(f->stack[f->count-count+1])) return "I require an integer array index";
    if (op==OP_ARR_SLICE && !array_index(f->stack[f->count-1])) return "I require an integer slice end";
    if (op==OP_ARR_PUSH || op==OP_ARR_SET) {
        if (!effect_value_matches(f->locals,f->stack[f->count-1],
                (NvmOwnershipEffectValue){element.tag,element.layout}))
            return "I require the exact array element type and ownership";
    }
    f->count-=count;
    if (op==OP_ARR_GET || op==OP_ARR_POP) type=element;
    return array_push_type(f,type)?NULL:"I cannot retain an array operation result";
}
