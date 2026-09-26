from pathlib import Path
import re,hashlib,json
p=Path('src_nano/typecheck.nano');s=p.read_text();original=s
names=['resource_type_identity','resource_concrete_identity','resource_concrete_payload_fact','resource_classify','resource_instantiated_classify','tc_selected_payload_union']
proof={}
for name in names:
 start=s.index('fn '+name+'(');end=s.index('\n}\n',start)+3
 function=s[start:end];signature,body=function.split('\n',1)
 params=signature.split('(',1)[1].split(') ->',1)[0]
 args=[v.split(':',1)[0].strip() for v in params.split(',')]
 assert args[0]=='parser'
 internal=signature.replace('fn '+name+'(','fn '+name+'_in_declarations(').replace('parser: Parser','declarations: ResourceDeclarations')+'\n'+body
 replacements={}
 for callee in names:replacements['('+callee+' parser']='('+callee+'_in_declarations declarations'
 replacements.update({'(parser_get_struct_def_count parser)':'(list_ASTStruct_length declarations.structs)','(parser_get_union_count parser)':'(list_ASTUnion_length declarations.unions)','(parser_get_struct_def parser ':'(list_ASTStruct_get declarations.structs ','(parser_get_union parser ':'(list_ASTUnion_get declarations.unions '})
 for before,after in replacements.items():internal=internal.replace(before,after)
 assert not re.search(r'\bparser\b',internal),name
 wrapper=signature+'\n    let declarations: ResourceDeclarations = ResourceDeclarations { structs: parser.structs, unions: parser.unions }\n    return ('+name+'_in_declarations declarations'+(' '+' '.join(args[1:]) if args[1:] else '')+')\n}\n'
 restored=internal.replace('fn '+name+'_in_declarations(','fn '+name+'(').replace('declarations: ResourceDeclarations','parser: Parser')
 for before,after in reversed(list(replacements.items())):restored=restored.replace(after,before)
 assert restored==function
 proof[name]={'original_sha256':hashlib.sha256(function.encode()).hexdigest(),'internal_sha256':hashlib.sha256(internal.encode()).hexdigest(),'inverse_mechanical_transform_equal':True}
 s=s[:start]+wrapper+'\n'+internal+s[end:]
needle='fn resource_type_identity('
pos=s.index(needle);s=s[:pos]+'''# I pass only these declaration handles through recursive classification.
struct ResourceDeclarations {
    structs: List<ASTStruct>,
    unions: List<ASTUnion>
}

'''+s[pos:]
checks={
'resource_type_identity':'assert (== (resource_type_identity_in_declarations declarations "Handle" 1) "#record:0")',
'resource_concrete_identity':'assert (== (resource_concrete_identity_in_declarations declarations "Box<Handle>" 1 0) "#union:0<#record:0>")',
'resource_concrete_payload_fact':'assert (resource_concrete_payload_fact_in_declarations declarations "Box<Handle>" 1 [] ["#record:0"] [] false [] 0)',
'resource_classify':'assert (resource_classify_in_declarations declarations "Handle" false)',
'resource_instantiated_classify':'assert (resource_instantiated_classify_in_declarations declarations "Box<Handle>" 1 0 false)',
 'tc_selected_payload_union':'assert (== (tc_selected_payload_union_in_declarations declarations "Box<int>.Some") 0)'}
for name,check in checks.items():
 s+='\nshadow '+name+'_in_declarations {\n    let parser: Parser = (resource_test_parser "resource struct Handle { fd: int } union Box<T> { Some { value: T }, None {} }")\n    let declarations: ResourceDeclarations = ResourceDeclarations { structs: parser.structs, unions: parser.unions }\n    '+check+'\n}\n'
old_shadows=re.findall(r'^shadow \w+ \{\n.*?^\}',original,re.M|re.S)
new_shadows=re.findall(r'^shadow \w+ \{\n.*?^\}',s,re.M|re.S)
assert new_shadows[:len(old_shadows)]==old_shadows
p.write_text(s)
Path('/tmp/resource-view-transform-proof.json').write_text(json.dumps({'functions':proof,'original_shadow_count':len(old_shadows),'added_shadow_count':len(new_shadows)-len(old_shadows),'original_shadow_bytes_unchanged':True},indent=2)+'\n')
print(json.dumps(proof,indent=2))
