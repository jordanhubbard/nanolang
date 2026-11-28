"""I compare complete classifier join states with unchanged C operations."""
from pathlib import Path
import hashlib, itertools, json, os, shlex, struct, tempfile, unittest
from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import native_link_flags
from tests.test_nanolang_translator_representation_stack import (
    actual_helpers as stack_helpers, operation, program, packet, grouped,
)

ROOT = Path(__file__).resolve().parents[1]

def raw(kind=2, tags=0, origin=-1, shape=0, payload=0, fields=None):
    return operation(15, kind, origin, shape, tags, payload, fields)

def join(target=30): return operation(10, parameter=target)
def pop(): return operation(2)
def plan(tags, count=1, target=30, function=7):
    return operation(12, origin=count, shape=function, tags=tags, parameter=target)
def graph_new(kind): return operation(16, kind=kind)
def graph_op(code, source, target=0, tags=0):
    return operation(code, shape=source, tags=tags, parameter=target)

def helpers(source):
    actual, old_spans = stack_helpers(source)
    spans = [
        (b'static int optional_scalar_tags(', b'#include "nvm2c_effect_flow.h"'),
        (b'static int variant_scalar_kind(', b'/* I share physical 64-bit slots'),
        (b'static int merge_record_join_field(', b'static int boolean_result('),
        (b'static uint8_t *sim_fields(', b'static int merge_fact('),
        (b'static int shape_record_return(', b'static int sim_push_slot('),
        (b'typedef struct {\n    Nvm2cSimSlot *slots;', b'static int sim_join('),
        (b'static int sim_join(', b'\nstatic int jump_target('),
    ]
    begin = b'typedef struct {\n    uint8_t *parameters;'
    end = b'\nenum {\n    NVM2C_GLOBAL_UNSTORED'
    assert source.count(begin) == source.count(end) == 1
    actual += source[source.index(begin):source.index(end)]
    for a, b in spans:
        assert source.count(a) == 1 and source.count(b) == (2 if a == b'static int sim_join(' else 1), (a, b)
        start = source.index(a)
        actual += source[start:source.index(b, start)]
    assert actual.count(b'static ') == 26
    a = source.index(b'typedef struct Nvm2cFieldBlock')
    b = source.index(b'#define NVM2C_SCALAR_UNKNOWN', a)
    join_body = source[source.index(spans[-1][0]):source.index(spans[-1][1])]
    assert hashlib.sha256(join_body).hexdigest() == '81432e73c0f4be78b2c104fcf1d5fdadefb9af604cd0dba5d85e74ffd6a019af'
    return actual, source[a:b], old_spans + [[a.decode(), b.decode()] for a,b in spans]

class RepresentationJoins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not source_compiler_command():
            raise RuntimeError('I require an explicitly selected genuine qualified NanoLang compiler.')
        cls.workspace = tempfile.TemporaryDirectory(prefix='nano-representation-joins-')
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory = Path(cls.workspace.name)
        source = (ROOT/'src/nanoisa/nvm2c.c').read_bytes()
        actual, types, spans = helpers(source)
        (cls.directory/'nvm2c_representation_joins_actual.h').write_bytes(actual)
        (cls.directory/'nvm2c_representation_joins_types.h').write_bytes(types)
        (cls.directory/'reference-source.json').write_text(json.dumps(dict(
            source='src/nanoisa/nvm2c.c', source_sha256=hashlib.sha256(source).hexdigest(),
            helper_sha256=hashlib.sha256(actual).hexdigest(), verbatim_helpers=26,
            types_sha256=hashlib.sha256(types).hexdigest(), spans=spans,
            minimal_owner_not_binary_layout=True, product_source_changed=False), indent=2)+'\n')
        cls.references = []
        for name, flags in [('ordinary',['-O2']), ('sanitized',[
                '-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            executable = cls.directory/('reference-'+name)
            checked([*shlex.split(os.environ.get('CC','cc')), '-std=c11','-Wall','-Wextra','-Werror',
                '-D_POSIX_C_SOURCE=200809L', *flags, '-I', cls.directory,
                ROOT/'tests/nvm2c_representation_joins_reference.c', ROOT/'src/nanoisa/isa.c',
                *native_link_flags(), '-o', executable], build=True)
            cls.references.append((name,[executable]))
        cls.routes = build_routes(ROOT/'tests/nvm2c_representation_joins_driver.nano', cls.directory)
        assert len(cls.routes) == 5

    def compare(self, rows, label, *, valid=True):
        paths = []; manifest = dict(qualification=False, whole_release=False, cases=[], routes=[])
        expected_programs = 0
        for name, data in rows:
            path = self.directory/(label+'-'+name+'.bin'); path.write_bytes(data); paths.append(path)
            manifest['cases'].append(dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            if valid: expected_programs += struct.unpack_from('<I', data)[0]
        baseline = None
        for name, command in [*self.references, *self.routes]:
            text = ''.join(checked([*command,*paths[i:i+8]],timeout=60) for i in range(0,len(paths),8))
            self.assertNotIn('ADAPTER_INPUT', text, name)
            if valid:
                self.assertNotIn('INVALID', text, name)
                self.assertEqual(sum(line.startswith('Z|') for line in text.splitlines()), expected_programs, name)
            if baseline is None: baseline = text
            else: self.assertEqual(text, baseline, name)
            data = text.encode(); (self.directory/(label+'-'+name+'.reports')).write_bytes(data)
            manifest['routes'].append(dict(name=name, sha256=hashlib.sha256(data).hexdigest(), lines=len(text.splitlines())))
        (self.directory/(label+'-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
        return baseline

    def test_named_plans_provenance_origins_and_partial_refusals(self):
        rows = [
            ('forward', packet([program([raw(1),join(),pop(),raw(8),join()],mode=1)])),
            ('backward', packet([program([raw(1),join(29),pop(),raw(8),join(29)],mode=1)])),
            ('finite', packet([program([raw(8,16386),join(),pop(),raw(2,32768),join()],mode=1)])),
            ('reset', packet([program([plan(3),raw(8,32771),join()],mode=1)])),
            ('stable', packet([program([plan(3),raw(8,3),join(),operation(11,parameter=1),pop(),raw(8,5),join()],mode=1)])),
            ('unchanged', packet([program([raw(8,3),join(),operation(11,parameter=1),join()],mode=1)])),
            ('height', packet([program([plan(3,2),raw(0,2),join()],mode=1)])),
            ('unknown', packet([program([raw(2,32768,2,7,13),join(),pop(),raw(0,2,3,9,14),join()],mode=1)])),
            ('partial', packet([program([raw(0,2,1,payload=13),raw(1,32,1),raw(0,2,1,payload=13),join(),pop(),pop(),pop(),raw(0,2,2,payload=14),raw(11,8,2),raw(9,16,2,payload=14),join()],capacity=3,mode=1)])),
            ('initial-error-empty', packet([program([raw(0,2),join(),pop(),join()],flags=1,mode=1)])),
            ('duplicate-plans', packet([program([plan(17,0,target=29,function=8),plan(3,1),plan(0,0),join()],mode=1)])),
        ]
        text = self.compare(rows,'named')
        for marker in [
            'I cannot yet widen tagged scalar storage on a backward stack edge in function 7',
            'I cannot join proved finite storage with unproved payload provenance',
            'I require stable scalar join facts before shape emission (function 7, target 30, slot 0, plan 0x3, actual 0x7)',
            'I found inconsistent scalar join height in function 7',
            'I found incompatible stack kinds at a join in function 7 after offset 29 (slot 1: 1 versus 11)',
            'P|0|7|30|1|0\n','P|0|7|30|1|3\n','T|0|0|-1|0|7|2|0|\n',
            'T|2|0|1|0|0|2|13|\n','J|1|0|1|0|0|0\n',
        ]: self.assertIn(marker,text)

    def test_every_kind_pair_and_every_scalar_mask(self):
        cases = [program([raw(a),join(),pop(),raw(b),join()],mode=1,defaults=(),locals=())
                 for a,b in itertools.product(range(256),repeat=2)]
        cases += [program([raw(8),join(),pop(),raw(8,mask),join()],mode=1,defaults=(),locals=())
                  for mask in range(65536)]
        masks = [0,1,2,3,5,6,7,9,10,11,17,32,33,2048,2049,16384,16386,16512,32768,32771,65535]
        for flags, planned, mask in itertools.product([0,24,32,64,128],masks,masks):
            cases.append(program([plan(planned),raw(8,mask),join(),join()],flags=flags,mode=1,defaults=(),locals=()))
        self.compare(grouped(cases,'matrix'),'matrix')

    def test_full_shapes_fields_cycles_and_directed_storage(self):
        cases = []
        representations = [0,1,2,3,4,5,6,7,8,9,11,13,14,16,65,96,255]
        for a,b,flags in itertools.product(representations,representations,[0,1,3,9,25,33]):
            cases.append(program([graph_new(0),graph_new(0),raw(a,0,0,1,fields=(2,65)),join(),pop(),
                raw(b,0,1,2,fields=(0,96)),join(),operation(9)],flags=flags,mode=1))
        cases += [program([graph_new(k),graph_new(k),raw(4,0,0,1,fields=(2,2)),join(),pop(),
            raw(4,0,1,2,fields=(0,8)),join(),operation(9)],flags=1,mode=1) for k in range(15)]
        cases += [program([graph_new(4),graph_op(17,1,0),graph_op(18,1,2),raw(4,0,0,1,fields=(2,)),join(),
            join(),operation(9)],flags=1,mode=1,defaults=(2,))]
        for opcode in [18,19,21,22,24,25,26]:
            cases.append(program([graph_new(0),graph_new(0),graph_op(opcode,1,2),raw(2,0,0,1),join(),pop(),
                raw(0,2,1,2),join(),operation(9)],flags=1,mode=1))
        self.compare(grouped(cases,'shapes'),'shapes')

    def test_complete_height_field_and_framing_frontiers(self):
        cases = [program([raw(0,2,origin),join(target)],function=function,argument=function,offset=offset,mode=1,defaults=(),locals=())
            for function,target,offset,origin in itertools.product([0,4294967295],[0,4294967295],[0,4294967295],[-2147483648,2147483647])]
        many = program([raw(0,2)]*65535+[join()],capacity=65535,mode=1,defaults=(),locals=())
        fields = bytes([2])*65535
        wide = program([raw(4,fields=fields),join(),pop(),raw(4,fields=bytes([0])*65535),join()],
                       mode=1,defaults=fields,locals=())
        rows = grouped(cases,'frontiers')+[('height65535',packet([many])),('fields65535',packet([wide]))]
        text = self.compare(rows,'frontiers')
        self.assertIn('J|1|65535|65535|0|0|0\n',text)
        self.assertIn('T|65534|0|-1|0|0|2|0|\n',text)
        self.assertIn('T|0|4|-1|1|0|0|0|'+','.join(['0']*65535)+'\n',text)
        good = packet([program([raw(0)],defaults=(),locals=())])
        bad = [b'',b'\0'*3,good+b'\0',b'\xff'*4,b'\0'*4194305]
        for index,value in [(6,255),(7,1),(25,2),(40,255),(42,2),(54,1)]:
            changed=bytearray(good);changed[index]=value;bad.append(bytes(changed))
        result = self.compare([(f'bad-{i}',data) for i,data in enumerate(bad)],'framing',valid=False)
        self.assertEqual(result,'INVALID\n'*len(bad))
