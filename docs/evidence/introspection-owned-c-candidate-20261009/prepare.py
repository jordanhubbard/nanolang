from pathlib import Path
import shutil
root=Path('/private/tmp/nanolang-match-guards-20261009'); prior=Path('/private/tmp/nanolang-introspection-c-callables-20261009');out=Path('/private/tmp/nanolang-introspection-owned-c-20261009')
for name in ['codegen.c','module.c','build-commands.txt','build.py']:
 s=(prior/name).read_text().replace(str(prior),str(out));(out/name).write_text(s)
p=out/'codegen.c';s=p.read_text();a=s.index('static void compile_module_names(');b=s.index('\n/* Handle ___module_',a)
s=s[:a]+'''static void compile_module_names(CG *cg, char **names, int count) {
    uint32_t *exits = count ? calloc((size_t)count, sizeof(*exits)) : NULL;
    if (count && !exits) { cg_error(cg, 0, "I could not allocate metadata index branches"); return; }
    for (int i = 0; i < count; ++i) {
        emit_op(cg, OP_DUP);
        emit_op(cg, OP_PUSH_I64, (int64_t)i);
        emit_op(cg, OP_I64_EQ);
        uint32_t next = emit_op(cg, OP_JMP_FALSE, (int32_t)0);
        emit_op(cg, OP_POP);
        emit_op(cg, OP_PUSH_STR, nvm_add_string(cg->module, names[i], (uint32_t)strlen(names[i])));
        exits[i] = emit_op(cg, OP_JMP, (int32_t)0);
        patch_jump(cg, next + 1, next, cg->code_size);
    }
    emit_op(cg, OP_POP);
    emit_op(cg, OP_PUSH_STR, nvm_add_string(cg->module, "", 0));
    for (int i = 0; i < count; ++i) patch_jump(cg, exits[i] + 1, exits[i], cg->code_size);
    free(exits);
}
'''+s[b:];p.write_text(s)
s=(root/'src/nanovirt/borrow_codegen.inc').read_text();marker='    if(n->type==AST_CALL && b->value_graph) {'
insert='''    if(n->type==AST_CALL && !n->as.call.func_expr && !n->as.call.borrow_mode &&
       n->as.call.name && !borrow_value_bound(b,n->as.call.name)) {
        uint16_t arity=0;
        uint8_t result=module_introspection_result(n->as.call.name,&arity);
        Function *declaration=env_get_function(b->env,n->as.call.name);
        if(result!=TAG_VOID && declaration && declaration->is_extern) {
            if(n->as.call.arg_count!=arity || (arity && borrow_expr(b,n->as.call.args[0])!=TAG_INT)) {
                borrow_error(b,n,"the exact module introspection arguments");return TAG_VOID;
            }
            if(!compile_module_introspection(cg,n->as.call.name)) {
                borrow_error(b,n,"a known module introspection operation");return TAG_VOID;
            }
            return result;
        }
    }
''';assert marker in s;s=s.replace(marker,insert+marker,1)
marker='        } else if(n->type==AST_FUNCTION && !n->as.function.is_extern) {'
insert='''        } else if(n->type==AST_FUNCTION && n->as.function.is_extern) {
            uint16_t arity=0;
            uint8_t result=module_introspection_result(n->as.function.name,&arity);
            if(result==TAG_VOID || n->as.function.param_count!=arity ||
               type_to_tag(n->as.function.return_type,n->as.function.return_struct_type_name,env)!=result ||
               (arity && n->as.function.params[0].type!=TYPE_INT))
                borrow_error(b,n,"the declared module introspection signature");
''';assert marker in s;s=s.replace(marker,insert+marker,1);(out/'borrow_codegen.inc').write_text(s)
