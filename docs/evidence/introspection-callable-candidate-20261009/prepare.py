from pathlib import Path
root=Path('/private/tmp/nanolang-match-guards-20261009')
out=Path('/private/tmp/nanolang-introspection-callables-20261009')
s=(root/'src_nano/compiler/nanoisa_codegen.nano').read_text()
s=s.replace('fn nisa_emit_module_fact(parser: Parser, call: ASTCall, name: string) -> string {','fn nisa_module_value(name: string, evaluated: string) -> string {',1)
a=s.index('            if (< kind 6) {',s.index('fn nisa_module_value('));b=s.index('            if (< kind 2) {',a)
s=s[:a]+'''            if (== kind 6) { return (+ evaluated (nisa_module_names info.functions)) }
            if (== kind 7) { return (+ evaluated (nisa_module_names info.structs)) }
'''+s[b:]
marker='shadow nisa_emit_module_fact {'
insert='''shadow nisa_module_value {
    let _: int = (nisa_reset)
    (mi_reset)
    assert (str_contains (nisa_module_value "___module_name_probe" "") "PUSH_STR")
    assert (str_starts_with (nisa_module_value "___module_function_name_missing" "  LOAD_LOCAL 0\\n") "  LOAD_LOCAL 0\\n")
    let _: int = (nisa_reset)
}

fn nisa_emit_module_fact(parser: Parser, call: ASTCall, name: string) -> string {
    let indexed: bool = (or (str_starts_with name "___module_function_name_") (str_starts_with name "___module_struct_name_"))
    if indexed {
        if (!= call.arg_count 1) { return (nisa_fail "I require the declared module introspection signature") }
        let arg: ASTStmtRef = (parser_get_call_arg parser call.arg_start)
        if (!= (nisa_expr_type parser arg.node_id arg.node_type) "int") { return (nisa_fail "I require an integer module introspection index") }
        return (nisa_module_value name (nisa_emit_expr parser arg.node_id arg.node_type))
    }
    if (!= call.arg_count 0) { return (nisa_fail "I require the declared module introspection signature") }
    return (nisa_module_value name "")
}

fn nisa_emit_module_function(parser: Parser, function: ASTFunction) -> string {
    let count: string = (int_to_string function.param_count)
    let mut argument: string = ""
    let mut types: array<string> = []
    let mut parameters: string = ""
    if (== function.param_count 1) {
        set argument (nisa_line_imm "LOAD_LOCAL" "0")
        set types ["int"]
        set parameters (+ ".parameters " (+ function.name " int\\n"))
    }
    set nisa_ordinary_functions (+ nisa_ordinary_functions (nisa_ordinary_contract parser function.return_type function.param_count types))
    set nisa_ordinary_count (+ nisa_ordinary_count 1)
    let header: string = (+ ".function " (+ function.name (+ " " (+ count (+ " " (+ count (+ " 0 " (+ function.return_type " 1\\n"))))))))
    return (+ header (+ (nisa_module_value function.name argument) (+ "  RET\\n.end\\n" parameters)))
}
shadow nisa_emit_module_function {
    let _: int = (nisa_reset)
    let tokens: List<LexerToken> = (tokenize_string "extern fn ___module_function_name_probe(index:int)->string" "module.nano" (diag_list_new))
    let parser: Parser = (parse_program tokens (list_LexerToken_length tokens) "module.nano")
    let text: string = (nisa_emit_module_function parser (parser_get_function parser 0))
    assert (str_contains text ".function ___module_function_name_probe 1 1 0 string 1")
    assert (str_contains text ".parameters ___module_function_name_probe int")
    let _: int = (nisa_reset)
}

'''
s=s.replace(marker,insert+marker,1)
s=s.replace('if (< function.body 0) { return (nisa_fail "I require a NanoISA body for a function value") }','if (and (< function.body 0) (== (nisa_module_result function.name) "")) { return (nisa_fail "I require a NanoISA body for a function value") }',1)
old='if (< function.body 0) { let _: int = (nisa_register_extern parser function) }\n        else { set out (+ out (nisa_emit_function parser function)) }'
new='''if (and (< function.body 0) (!= (nisa_module_result function.name) "")) { set out (+ out (nisa_emit_module_function parser function)) }
        else if (< function.body 0) { let _: int = (nisa_register_extern parser function) }
        else { set out (+ out (nisa_emit_function parser function)) }'''
assert old in s;s=s.replace(old,new,1)
old='''            if (< func.body 0) {
                if program { let _: int = (nisa_register_extern parser func) }
            } else {'''
new='''            if (and (< func.body 0) (!= (nisa_module_result func.name) "")) {
                set out (+ out (nisa_emit_module_function parser func))
                set emitted (+ emitted 1)
            } else if (< func.body 0) {
                if program { let _: int = (nisa_register_extern parser func) }
            } else {'''
assert old in s;s=s.replace(old,new,1)
(out/'nanoisa_codegen.nano').write_text(s)
s=(root/'src_nano/nanoc_v06.nano').read_text().replace('import "src_nano/compiler/nanoisa_codegen.nano"',f'import "{out}/nanoisa_codegen.nano"',1)
(out/'nanoc_v06.nano').write_text(s)
