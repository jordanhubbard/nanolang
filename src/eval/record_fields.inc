/* I share my actual record read and byte-context rules with the native unit. */
static bool eval_record_field_context(const StructDef *definition,
        const char *field, Value *value) {
    if (!definition) return true;
    for (int i = 0; i < definition->field_count; ++i) {
        if (definition->field_names[i][0] != field[0] || strcmp(definition->field_names[i], field)) continue;
        if (definition->field_types[i] != TYPE_U8) return true;
        if (value->type != VAL_INT) {
            fprintf(stderr, "I require a checked integer at a byte record field.\n");
            return false;
        }
        value->as.int_val = (int64_t)(uint8_t)value->as.int_val;
        return true;
    }
    return true;
}

static Value eval_record_field_value(const StructValue *record, const char *field) {
    for (int i = 0; i < record->field_count; ++i) {
        if (record->field_names[i][0] == field[0] && strcmp(record->field_names[i], field) == 0) {
            /* I return owned STRING storage and preserve other value identities. */
            if (record->field_values[i].type == VAL_STRING)
                return create_string(record->field_values[i].as.string_val);
            return record->field_values[i];
        }
    }
    fprintf(stderr, "Error: Struct '%s' has no field '%s'\n", record->struct_name, field);
    return create_void();
}
