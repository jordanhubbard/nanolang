/* I share actual owned record-literal cleanup with my boundary unit. */
static void discard_literal_record(StructValue *record) {
    if (!record) return;
    for (int i = 0; i < record->field_count; i++) {
        Value field = record->field_values[i];
        if (field.type == VAL_STRUCT) discard_literal_record(field.as.struct_val);
        else if (field.type == VAL_STRING) {
            if (gc_is_managed(field.as.string_val)) gc_release(field.as.string_val);
            else free(field.as.string_val);
        }
    }
    release_struct_carrier_names(record);
    free(record->field_values);
    free(record);
}

