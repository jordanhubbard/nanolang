static int flow_one(NvmShapeGraph *g, NvmShapeConversion conversion, int *changed, int final) {
    size_t count = 0, capacity = 0, cursor = 0;
    FlowPair *queue = grow(g, NULL, &capacity, 1, sizeof *queue);
    if (!queue) return 0;
    NvmShapeId producer = conversion.source;
    if (conversion.checked_payload) {
        NvmShapeKind kind = nvm_shape_kind(g, producer);
        if (kind == NVM_SHAPE_UNKNOWN) { free(queue); return !g->error; }
        if (kind == NVM_SHAPE_OPTIONAL) producer = nvm_shape_child(g, producer, 0);
    }
    queue[count++] = (FlowPair){producer, conversion.target,
        conversion.checked_payload, conversion.alias_view, 0};
    while (cursor < count && !g->error) {
        FlowPair pair = queue[cursor++];
        NvmShapeId source = nvm_shape_root(g, pair.source), target = nvm_shape_root(g, pair.target);
        if (!source || !target) break;
        if (source == target) continue;
        int seen = 0;
        for (size_t i = 0; i + 1 < cursor; ++i)
            if (nvm_shape_root(g, queue[i].source) == source &&
                nvm_shape_root(g, queue[i].target) == target && queue[i].exact == pair.exact && queue[i].alias_view == pair.alias_view &&
                queue[i].optional_payload == pair.optional_payload) seen = 1;
        if (seen) continue;
        uint16_t tags = g->nodes[source - 1].value_tags;
        if (tags & ~g->nodes[target - 1].value_tags) {
            g->nodes[target - 1].value_tags |= tags;
            *changed = 1;
        }
        NvmShapeKind from = g->nodes[source - 1].kind, to = g->nodes[target - 1].kind;
        if (from == NVM_SHAPE_UNKNOWN) {
            if (final && (to == NVM_SHAPE_VARIANT_SCALAR || to == NVM_SHAPE_VARIANT_INT_ARRAY)) {
                fail(g, "I require proved scalar producers for variant scalar storage"); break;
            }
            continue;
        }
        if (to == NVM_SHAPE_UNKNOWN) {
            if (!flow_kind(g, target, from, changed)) break;
            to = from;
        }
        /* Shared record-array handles retain caller and callee views. A
         * tagged field can flow back into an exact scalar view because native
         * projection checks its tag before use; I retain the payload constraint
         * rather than rewriting the caller's constructor as optional storage. */
        if (pair.alias_view && from == NVM_SHAPE_OPTIONAL &&
            (to == NVM_SHAPE_STRING || to == NVM_SHAPE_INT || to == NVM_SHAPE_BOOL ||
             to == NVM_SHAPE_FLOAT || to == NVM_SHAPE_ARRAY || to == NVM_SHAPE_MAP ||
             to == NVM_SHAPE_FUNCTION || to == NVM_SHAPE_U8 || to == NVM_SHAPE_BSTRING)) {
            NvmShapeId payload = nvm_shape_child(g, source, 0);
            FlowPair *next = grow(g, queue, &capacity, count + 1, sizeof *queue);
            if (!next || !payload) break;
            queue = next; queue[count++] = (FlowPair){payload, target, 1, pair.alias_view, 0};
            continue;
        }
        if (!pair.exact && from == NVM_SHAPE_OPTIONAL &&
            (to == NVM_SHAPE_STRING || to == NVM_SHAPE_INT || to == NVM_SHAPE_BOOL ||
             to == NVM_SHAPE_FLOAT)) {
            if (!g->nodes[target - 1].conversion_kind) {
                snprintf(g->error_detail, sizeof g->error_detail,
                         "I cannot widen an exactly constrained %s destination at nodes %u/%u (conversion %u/%u)",
                         kind_name(to), source, target, conversion.source, conversion.target);
                fail(g, g->error_detail); break;
            }
            /* I widen only inferred destination storage. Its old scalar kind
             * belongs to a fresh payload, never to the wrapper itself. */
            if (!flow_kind(g, target, NVM_SHAPE_OPTIONAL, changed)) break;
            NvmShapeId payload = nvm_shape_child(g, target, 0);
            if (!payload || !flow_kind(g, payload, to, changed)) break;
            to = NVM_SHAPE_OPTIONAL;
        }
        /* I retain a present field's payload when a checked shared-record
         * view boxes it, including below an optional ancestor. Exact storage
         * outside that alias view still forbids this injection. */
        if ((!pair.exact || pair.alias_view) && (from == NVM_SHAPE_STRING || from == NVM_SHAPE_INT ||
                            from == NVM_SHAPE_BOOL || from == NVM_SHAPE_FLOAT || from == NVM_SHAPE_ARRAY ||
                            from == NVM_SHAPE_MAP || from == NVM_SHAPE_FUNCTION || from == NVM_SHAPE_U8 || from == NVM_SHAPE_BSTRING) && to == NVM_SHAPE_OPTIONAL) {
            NvmShapeId payload = nvm_shape_child(g, target, 0);
            FlowPair *next = grow(g, queue, &capacity, count + 1, sizeof *queue);
            if (!next || !payload) break;
            queue = next; queue[count++] = (FlowPair){source, payload, 1, pair.alias_view, 0};
            continue;
        }
        /* I widen an inferred boxed payload to its producer's finite set.
         * A nested projection can first acquire INT from a sibling arm;
         * that is destination storage, not an exact scalar constraint.
         * Bare scalars, array elements and exact payloads remain exact. */
        if (pair.optional_payload && g->nodes[target - 1].conversion_kind &&
            from == NVM_SHAPE_VARIANT_SCALAR &&
            (to == NVM_SHAPE_INT || to == NVM_SHAPE_BOOL || to == NVM_SHAPE_FLOAT ||
             to == NVM_SHAPE_STRING || to == NVM_SHAPE_U8)) {
            if (!flow_kind(g, target, from, changed)) break;
            to = from;
        }
        /* An explicitly declared union destination accepts either exact
         * numeric member without changing the producer or OPTIONAL itself. */
        if (to == NVM_SHAPE_NUMERIC &&
            (from == NVM_SHAPE_INT || from == NVM_SHAPE_FLOAT)) continue;
        /* Only an explicitly seeded variant payload set accepts these
         * exact scalar producers. I do not change their source constraints. */
        if (to == NVM_SHAPE_VARIANT_SCALAR &&
            (from == NVM_SHAPE_INT || from == NVM_SHAPE_BOOL ||
             from == NVM_SHAPE_FLOAT || from == NVM_SHAPE_STRING || from == NVM_SHAPE_U8)) continue;
        /* I admit only a proved integer-array member, not any heap handle.
         * I wait for late element resolution without binding the producer. */
        if (to == NVM_SHAPE_VARIANT_INT_ARRAY) {
            if (from == NVM_SHAPE_INT || from == NVM_SHAPE_BOOL ||
                from == NVM_SHAPE_FLOAT || from == NVM_SHAPE_STRING || from == NVM_SHAPE_U8 ||
                from == NVM_SHAPE_VARIANT_SCALAR) continue;
            if (from == NVM_SHAPE_ARRAY) {
                NvmShapeId element = nvm_shape_lookup(g, source, 0);
                NvmShapeKind member = element ? nvm_shape_kind(g, element) : NVM_SHAPE_UNKNOWN;
                if (member == NVM_SHAPE_INT) continue;
                if (member == NVM_SHAPE_UNKNOWN && !final) continue;
                fail(g, "I require an exact integer element for variant array storage"); break;
            }
        }
        if (from != to) {
            snprintf(g->error_detail, sizeof g->error_detail,
                     "I cannot convert aggregate storage %s to %s at nodes %u/%u",
                     kind_name(from), kind_name(to), source, target);
            fail(g, g->error_detail); break;
        }
        if (from == NVM_SHAPE_ARRAY && !nvm_shape_array_alias(g, source, target)) break;
        if (from == NVM_SHAPE_FUNCTION) {
            NvmShapeNode *producer = &g->nodes[source - 1], *storage = &g->nodes[target - 1];
            for (size_t i = 0; i < producer->function_count && !g->error; ++i)
                function_add(g, storage, producer->functions[i], changed);
        }
        size_t edges = g->nodes[source - 1].count;
        for (size_t i = 0; i < edges && !g->error; ++i) {
            ShapeEdge edge = g->nodes[source - 1].edges[i];
            NvmShapeId child_source = nvm_shape_root(g, edge.child);
            NvmShapeId child_target = nvm_shape_lookup(g, target, edge.index);
            if (!child_target && !g->error) {
                /* I preserve cycles and sharing when creating missing target
                 * edges, without equating an existing destination view. */
                NvmShapeKind child_kind = nvm_shape_kind(g, child_source);
                for (size_t j = 0; j < cursor; ++j)
                    if ((child_kind == NVM_SHAPE_RECORD || child_kind == NVM_SHAPE_ARRAY ||
                         child_kind == NVM_SHAPE_MAP || child_kind == NVM_SHAPE_OPTIONAL) &&
                        nvm_shape_root(g, queue[j].source) == child_source &&
                        nvm_shape_kind(g, queue[j].target) == child_kind) {
                        child_target = nvm_shape_root(g, queue[j].target); break;
                    }
                if (!child_target) child_target = nvm_shape_new(g, NVM_SHAPE_UNKNOWN);
                if (!child_target) break;
                NvmShapeNode *node = &g->nodes[target - 1];
                ShapeEdge *next = grow(g, node->edges, &node->capacity, node->count + 1, sizeof *next);
                if (!next) break;
                node->edges = next; node->edges[node->count++] = (ShapeEdge){edge.index, child_target};
                *changed = 1;
            }
            FlowPair *next = grow(g, queue, &capacity, count + 1, sizeof *queue);
            if (!next) break;
            queue = next;
            /* Record arrays nested inside copied records still share their
             * element handles. Their projections retain checked field views. */
            queue[count++] = (FlowPair){child_source, child_target,
                pair.exact || from == NVM_SHAPE_OPTIONAL || from == NVM_SHAPE_MAP,
                pair.alias_view || (from == NVM_SHAPE_ARRAY &&
                    nvm_shape_kind(g, child_source) == NVM_SHAPE_RECORD),
                from == NVM_SHAPE_OPTIONAL};
        }
    }
    free(queue);
    return !g->error;
}

