from pathlib import Path
import shutil,difflib
root=Path('/private/tmp/nanolang-match-guards-20261009'); work=Path(__file__).resolve().parent; overlay=work/'candidate'
shutil.copytree(root/'src_nano',overlay/'src_nano',dirs_exist_ok=True)
for name in ('src','include','modules','std','stdlib','bin','lib'):
 path=overlay/name
 if not path.exists():path.symlink_to(root/name,target_is_directory=True)
p=overlay/'src_nano/compiler/module_bindings.nano';original=p.read_text();s=original
s=s.replace('let mut mb_enabled: bool = false','let mut mb_binding_indices: HashMap<string, int> = (map_new)\nlet mut mb_enabled: bool = false',1)
s=s.replace('    set mb_global_import_owners []','    set mb_binding_indices (map_new)\n    set mb_global_import_owners []',1)
lookup='''fn mb_binding_key(owner: int, name: string) -> string {
    return (+ (int_to_string owner) (+ ":" name))
}
shadow mb_binding_key {
    assert (!= (mb_binding_key 1 "2:x") (mb_binding_key 12 "x"))
    assert (!= (mb_binding_key -1 "x") (mb_binding_key 1 "x"))
    assert (== (mb_binding_key 0 "") "0:")
}

fn mb_binding_index(owner: int, name: string) -> int {
    let key: string = (mb_binding_key owner name)
    if (map_has mb_binding_indices key) { return (map_get mb_binding_indices key) }
    return -1
}
shadow mb_binding_index {
    (mb_reset [1, 8])
    assert (== (mb_binding_index 0 "answer") -1)
    (mb_add_target 0 "answer" "first" 1)
    (mb_add_target 1 "answer" "second" 0)
    assert (== (mb_binding_index 0 "answer") 0)
    assert (== (mb_binding_index 1 "answer") 1)
    (mb_add_target 0 "answer" "ignored" 0)
    assert (== (mb_lookup 0 "answer") "first")
    assert (== (mb_target_owner 0 "answer") 1)
    (mb_reset [1])
    assert (== (mb_binding_index 0 "answer") -1)
    (mb_add_target 0 "answer" "fresh" 0)
    assert (== (mb_lookup 0 "answer") "fresh")
    assert (== (mb_target_owner 0 "answer") 0)
    (mb_reset [])
}

'''
s=s.replace('pub fn mb_add_target(',lookup+'pub fn mb_add_target(',1)
old='''    let mut i: int = 0
    while (< i (array_length mb_names)) {
        if (and (== (at mb_owners i) owner) (== (at mb_names i) name)) { return }
        set i (+ i 1)
    }
'''
assert old in s;s=s.replace(old,'    if (>= (mb_binding_index owner name) 0) { return }\n    (map_put mb_binding_indices (mb_binding_key owner name) (array_length mb_names))\n',1)
for func,ret,fallback in [('mb_lookup','mb_targets','name'),('mb_target_owner','mb_definition_owners','owner')]:
 start=s.index('pub fn '+func+'(');end=s.index('\nshadow '+func,start)
 region=s[start:end]
 loop=region[region.index('    let mut i: int = 0'):region.index('    return '+fallback+';') if '    return '+fallback+';' in region else region.rindex('    return '+fallback)]
 replacement=f'    let index: int = (mb_binding_index owner name)\n    if (>= index 0) {{ return (at {ret} index) }}\n'
 s=s[:start]+region.replace(loop,replacement)+s[end:]
p.write_text(s)
(work/'module-bindings-candidate.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/src_nano/compiler/module_bindings.nano',tofile='b/src_nano/compiler/module_bindings.nano')))
