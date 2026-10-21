"""I qualify complete paired map publication, with explicit producer identities."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import struct
import sys
import tempfile
import unittest
from tests.native_process import run_captured, native_compile_timeout
from tests import test_record_map_native as native

ROOT = Path(__file__).resolve().parents[1]


class MapFieldPublication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-map-field-publication-'))
        print('I retain paired map publisher evidence at', cls.work, flush=True)
        cls.status = []
        cls.tools = {name: Path(os.environ.get('MAP_PUBLICATION_'+name.upper(), str(ROOT/'bin'/name)))
                     for name in ('nano_virt', 'nanoisa', 'nano_vm', 'nvm2c')}
        cls.c_shadow = Path(os.environ.get('MAP_PUBLICATION_C_SHADOW', str(ROOT/'obj/borrow_shadow_names')))
        cls.compilers = [Path(p) for p in shlex.split(os.environ.get('MAP_PUBLICATION_COMPILERS',
                          str(ROOT/'bin/nanoc_stage1')+' '+str(ROOT/'bin/nanoc_stage2')))]
        cls.cc = shlex.split(os.environ.get('MAP_PUBLICATION_CC', 'cc'))
        cls.env = dict(os.environ, LSAN_OPTIONS='')
        if sys.platform.startswith('linux'):
            cls.env.setdefault('NANO_AS_CAPTURE_HELPER', str(ROOT/'bin/nano_as_capture.so'))
            if not Path(cls.env['NANO_AS_CAPTURE_HELPER']).is_file():
                raise AssertionError('I require the selected real Linux assembler capture helper')
        cls.driver_source = cls.work/'raw-shadow.nano'
        cls.driver_source.write_bytes((ROOT/'tests/nanoisa/fixtures/map_publication_driver.nano.txt').read_bytes())
        selected = os.environ.get('MAP_PUBLICATION_RAW_DRIVERS')
        cls.raw_drivers = [Path(p) for p in shlex.split(selected)] if selected else []
        if selected is None:
            for i, compiler in enumerate([ROOT/'bin/nanoc_c', *cls.compilers]):
                output = cls.work/f'raw-driver-{i}'
                cls.command('raw-build-'+str(i), [compiler, cls.driver_source, '-o', output], build=True)
                cls.raw_drivers.append(output)
        if not cls.compilers or not cls.raw_drivers:
            raise AssertionError('I require explicit checked and raw Nano producers')
        cls.pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [*cls.tools.values(), cls.c_shadow, *cls.compilers, *cls.raw_drivers]}
        if 'NANO_AS_CAPTURE_HELPER' in cls.env:
            helper = cls.env['NANO_AS_CAPTURE_HELPER']
            cls.pins[helper] = hashlib.sha256(Path(helper).read_bytes()).hexdigest()
        (cls.work/'provider-pins.json').write_text(json.dumps(cls.pins, indent=2)+'\n')
        # I retain actual checked shadow bytes and exec the same VM with identical arguments.
        cls.capture = cls.work/'capture-vm'
        cls.capture.write_text('#!'+sys.executable+'\n'+'''import hashlib,json,os,sys
from pathlib import Path
args=sys.argv[1:]
assert len(args)==2 and args[0]=='--check-shadows'
p=Path(os.environ['MAP_SHADOW_CAPTURE']);assert not p.exists()
data=Path(args[1]).read_bytes();p.write_bytes(data)
p.with_suffix('.json').write_text(json.dumps(dict(argv=args,sha256=hashlib.sha256(data).hexdigest(),vm=os.environ['MAP_REAL_VM'])))
os.execv(os.environ['MAP_REAL_VM'],[os.environ['MAP_REAL_VM']]+args)
''')
        cls.capture.chmod(0o755)

    @classmethod
    def command(cls, name, args, *, build=False, env=None, fail=False):
        args = list(map(str, args))
        budget = native_compile_timeout(default=900 if name.startswith('raw-build-') else 600) if build else 180
        p = run_captured(args, cwd=ROOT, env=env or cls.env, timeout=budget)
        (cls.work/(name+'.stdout')).write_bytes(p.stdout)
        (cls.work/(name+'.stderr')).write_bytes(p.stderr)
        cls.status.append(dict(name=name, argv=args, status=p.returncode, timeout=budget))
        (cls.work/'status.json').write_text(json.dumps(cls.status, indent=2)+'\n')
        if (p.returncode == 0) == fail:
            raise AssertionError(f'{name}: {p.returncode}\n{p.stdout.decode(errors="replace")}\n{p.stderr.decode(errors="replace")}')
        for marker in (b'AddressSanitizer', b'UndefinedBehaviorSanitizer', b'runtime error:'):
            if marker in p.stderr:
                raise AssertionError((name, p.stderr))
        return p.stdout

    @classmethod
    def tearDownClass(cls):
        after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in cls.pins}
        (cls.work/'provider-pins-after.json').write_text(json.dumps(after, indent=2)+'\n')
        if after != cls.pins:
            raise AssertionError('I changed a selected provider during qualification')

    @staticmethod
    def facts(text, kind):
        return bytes.fromhex(''.join(re.findall(r'^\.'+kind+r' "([0-9a-f]*)"$', text, re.M)))

    def check_wire(self, text, published):
        layouts, owned = [self.facts(text, k) for k in ('layouts', 'ownership')]
        if not published:
            self.assertEqual((layouts, owned), (b'', b''))
            return None
        version, count = struct.unpack_from('<II', owned)
        self.assertEqual(version, 3)
        self.assertEqual(struct.unpack_from('<I', layouts)[0], count)
        field_maps = []; at = 4
        for layout in range(count):
            kind, reserved, fields, name = struct.unpack_from('<BBHI', layouts, at); at += 8
            self.assertEqual((kind, reserved, name), (0, 0, 0xffffffff))
            for field in range(fields):
                tag, reserved, child, name = struct.unpack_from('<B3sII', layouts, at); at += 12
                self.assertEqual(reserved, b'\0'*3)
                if tag == 13:
                    self.assertEqual((child, name), (0xffffffff, 0xffffffff))
                    field_maps.append((layout, field, 0, 0))
        self.assertEqual(at, len(layouts))
        at = 8+((count+3)&~3)
        functions, = struct.unpack_from('<I', owned, at); at += 4
        for _ in range(functions):
            locals_, arity = struct.unpack_from('<HH', owned, at)
            self.assertLessEqual(arity, locals_)
            at += 4+8*(locals_+1)
        length, = struct.unpack_from('<I', owned, at); at += 4
        self.assertEqual(owned[at:at+length], b'\0'*4); at += length
        extensions, = struct.unpack_from('<I', owned, at); at += 4
        rows = {}
        for _ in range(extensions):
            kind, revision, length = struct.unpack_from('<HHI', owned, at); at += 8
            self.assertEqual(revision, 1); self.assertNotIn(kind, rows)
            rows[kind] = owned[at:at+length]; at += length
        self.assertEqual(at, len(owned)); self.assertIn(list(rows), ([5], [2, 5]))
        table = rows[5]
        self.assertEqual(table[:32], struct.pack('<IB3xIB3xIIII', 2, 1, 0xffffffff, 5, 0xffffffff, 1, 1, 0))
        bindings, = struct.unpack_from('<I', table, 32)
        self.assertEqual(len(table), 36+12*bindings)
        self.assertEqual([struct.unpack_from('<IHHI', table, 36+12*i) for i in range(bindings)], field_maps)
        self.assertTrue(field_maps)
        return layouts, rows

    def qualify(self, name, module, published):
        self.command(name+'-verify', [self.tools['nano_vm'], '--verify-only', module])
        self.command(name+'-vm', [self.tools['nano_vm'], module])
        text = self.command(name+'-dump', [self.tools['nanoisa'], 'dump', module]).decode()
        wire = self.check_wire(text, published)
        asm = self.work/(name+'.nasm'); copy = self.work/(name+'-roundtrip.nvm'); asm.write_text(text)
        self.command(name+'-assemble', [self.tools['nanoisa'], 'asm', asm, '-o', copy])
        self.assertEqual(text, self.command(name+'-roundtrip', [self.tools['nanoisa'], 'dump', copy]).decode())
        c = self.work/(name+'.c')
        self.command(name+'-native', [self.tools['nvm2c'], module, '-o', c])
        contents = c.read_text()
        if published:
            contents = '#define main fixture_main\n'+native.RecordMapNative.force_collection(contents)+'''
#undef main
int main(void){(void)nmap_collection_budget;int r=fixture_main();nmap_collect();if(nmap_owned_head||nrec_owned_head||nmap_live_bytes||nagg_live_bytes||nroot_head)abort();return r;}
'''
            c.write_text(contents)
        for opt in ('0', '2'):
            exe = self.work/(name+'-O'+opt)
            self.command(name+'-build-O'+opt, [*self.cc, '-std=c11', '-O'+opt, '-Wall', '-Wextra', '-Werror',
                         '-fsanitize=address,undefined', '-fno-sanitize-recover=all', c, '-lm', '-o', exe], build=True)
            self.command(name+'-run-O'+opt, [exe])
        return wire

    def corpus(self, label, source, ordinary=True, shadow=True):
        path = self.work/(label+'.nano'); path.write_text(source)
        wires = []; shadow_wires = []
        seed = self.work/(label+'-seed.nvm')
        self.command(label+'-seed-source', [self.tools['nano_virt'], path, '--emit-nvm', '-o', seed])
        wires.append(self.qualify(label+'-seed', seed, ordinary))
        name = label+'-seed-shadow'; asm = self.work/(name+'.nasm'); module = self.work/(name+'.nvm')
        asm.write_bytes(self.command(name+'-source', [self.c_shadow, path]))
        self.command(name+'-assemble', [self.tools['nanoisa'], 'asm', asm, '-o', module])
        shadow_wires.append(self.qualify(name, module, shadow))
        for i, compiler in enumerate(self.compilers):
            name = f'{label}-checked-{i}'; module = self.work/(name+'.nvm'); selected = self.work/(name+'-shadow.nvm')
            env = dict(self.env, NANO_VM=str(self.capture), MAP_REAL_VM=str(self.tools['nano_vm']), MAP_SHADOW_CAPTURE=str(selected))
            self.command(name+'-source', [compiler, path, '--emit-nvm', '-o', module], env=env)
            wires.append(self.qualify(name, module, ordinary))
            shadow_wires.append(self.qualify(name+'-shadow', selected, shadow))
        for i, driver in enumerate(self.raw_drivers):
            name = f'{label}-raw-{i}'; asm = self.work/(name+'.nasm'); module = self.work/(name+'.nvm')
            asm.write_bytes(self.command(name+'-source', [driver, path, 0, 'raw']))
            self.command(name+'-assemble', [self.tools['nanoisa'], 'asm', asm, '-o', module])
            shadow_wires.append(self.qualify(name, module, shadow))
            name += '-program'
            asm = self.work/(name+'.nasm'); module = self.work/(name+'.nvm')
            asm.write_bytes(self.command(name+'-source', [driver, path, 0, 'program']))
            self.command(name+'-assemble', [self.tools['nanoisa'], 'asm', asm, '-o', module])
            wires.append(self.qualify(name, module, ordinary))
        self.assertTrue(all(wire == wires[0] for wire in wires))
        self.assertTrue(all(wire == shadow_wires[0] for wire in shadow_wires))

    def test_unchanged_and_concat(self):
        source = (ROOT/'tests/nanoisa/fixtures/map_record_fields.nano').read_text()
        self.corpus('unchanged', source)
        self.corpus('concat', source.replace('(remember alias "second" 2)', '(remember alias (+ "sec" "ond") 2)'))

    def test_exact_integer_delete(self):
        source = (ROOT/'tests/unit/test_map_field_operations.nano').read_text()
        self.assertEqual(source.count('HashMap<string,string>'), 1)
        self.assertEqual(source.count('"kept"'), 2)
        source = source.replace('HashMap<string,string>', 'HashMap<string,int>').replace('"kept"', '42')
        self.corpus('integer-delete', source)

    def test_map_only_multiple_fields_forward_record(self):
        self.corpus('map-only-forward', '''struct Root { child: Leaf }
struct Leaf { first: HashMap<string,int>, second: HashMap<string,int> }
fn main() -> int {
    let first: HashMap<string,int> = (map_new)
    let second: HashMap<string,int> = (map_new)
    let leaf: Leaf = Leaf { first: first, second: second }
    let root: Root = Root { child: leaf }
    let alias: Root = root
    (map_put alias.child.first "first" 7)
    (map_put alias.child.second "second" 9)
    assert (== (map_get root.child.first "first") 7)
    assert (== (map_get root.child.second "second") 9)
    assert (== (map_length root.child.first) 1)
    assert (== (map_length root.child.second) 1)
    assert (not (map_has root.child.first "second"))
    assert (not (map_has root.child.second "first"))
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_nested_u8(self):
        self.corpus('nested-u8', (ROOT/'tests/nanoisa/fixtures/map_nested_u8_borrowed.nano').read_text())

    def test_independent_shadow_omission(self):
        source = (ROOT/'tests/nanoisa/fixtures/map_record_fields.nano').read_text()
        original = 'shadow main { assert (== (main) 0) }'
        self.assertEqual(source.count(original), 1)
        self.corpus('shadow-omit', source.replace(original, 'shadow main { assert (== (main) 0) assert (== (str_trim " x ") "x") }'), shadow=False)

    def test_mixed_omission(self):
        source = (ROOT/'tests/nanoisa/fixtures/map_record_fields.nano').read_text()
        self.assertEqual(source.count('    return 0'), 1)
        self.corpus('mixed-omit', source.replace('    return 0',
            '    (map_remove values "missing")\n    assert (== (str_trim " x ") "x")\n    return 0'), ordinary=False, shadow=False)

    def test_raw_declaration_mismatch_has_no_authority(self):
        source = self.work/'declaration-mismatch.nano'
        source.write_text('''struct Box { values: HashMap<string,int> }
fn main() -> int {
    let values: HashMap<string,string> = (map_new)
    (map_put values "key" "value")
    let box: Box = Box { values: values }
    return 0
}
shadow main { assert (== (main) 0) }
''')
        for i, compiler in enumerate([self.tools['nano_virt'], *self.compilers]):
            output = self.work/f'mismatch-checked-{i}.nvm'
            output.write_bytes(b'prior-output')
            self.command(f'mismatch-checked-{i}', [compiler, source, '--emit-nvm', '-o', output], fail=True)
            self.assertEqual(output.read_bytes(), b'prior-output')
        for i, driver in enumerate(self.raw_drivers):
            for mode in ('program', 'raw'):
                text = self.command(f'mismatch-raw-{i}-{mode}', [driver, source, 0, mode], fail=True)
                self.assertIn(b'I require the declared type for record field values', text)
                self.assertNotIn(b'.ownership ', text)
                self.assertNotIn(b'.layouts ', text)

    def test_malformed_wire_preserves_output(self):
        source = self.work/'malformed-source.nano'
        source.write_bytes((ROOT/'tests/nanoisa/fixtures/map_record_fields.nano').read_bytes())
        module = self.work/'malformed-base.nvm'
        self.command('malformed-base-source', [self.tools['nano_virt'], source, '--emit-nvm', '-o', module])
        text = self.command('malformed-base-dump', [self.tools['nanoisa'], 'dump', module]).decode()
        self.check_wire(text, True)
        owned = bytearray(self.facts(text, 'ownership')); owned[-1] = 255
        bad = re.sub(r'^\.ownership "[0-9a-f]*"\n', '', text, flags=re.M)+'\n.ownership "'+owned.hex()+'"\n'
        assembly = self.work/'malformed.nasm'; assembly.write_text(bad)
        output = self.work/'malformed-output.nvm'; output.write_bytes(b'prior-output')
        self.command('malformed-refusal', [self.tools['nanoisa'], 'asm', assembly, '-o', output], fail=True)
        self.assertEqual(output.read_bytes(), b'prior-output')
        assembly.write_text(text)
        self.command('malformed-retry', [self.tools['nanoisa'], 'asm', assembly, '-o', output])
        self.command('malformed-retry-verify', [self.tools['nano_vm'], '--verify-only', output])
