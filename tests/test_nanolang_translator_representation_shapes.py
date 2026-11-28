"""I compare all logical representation graph adapters with actual unchanged C."""
from pathlib import Path
import hashlib,json,os,shlex,struct,tempfile,unittest,itertools
from tests.nanolang_translator_routes import build_routes,checked,source_compiler_command
from tests.native_process import native_link_flags
ROOT=Path(__file__).resolve().parents[1]

def operation(code,a=0,b=0,c=0,fields=None):
    data=struct.pack('<BIII',code,a,b,c)
    if code==18:
        left,right=fields;assert len(left)==len(right)
        data+=struct.pack('<H',len(left))+bytes(left)+bytes(right)
    return data

def program(steps,flags=1,mode=0,function=17,offset=29,opcode=32):
    return struct.pack('<4BIII',mode,flags,opcode,0,function,offset,len(steps))+b''.join(steps)

def packet(programs):return struct.pack('<I',len(programs))+b''.join(programs)

def grouped(programs,prefix='rows',limit=256):
    result=[];rows=[];size=4
    for data in programs:
        if rows and (len(rows)>=limit or size+len(data)>1048576):
            result.append((f'{prefix}-{len(result):04d}',packet(rows)));rows=[];size=4
        assert len(data)+4<=1048576
        rows.append(data);size+=len(data)
    if rows:result.append((f'{prefix}-{len(result):04d}',packet(rows)))
    return result

def payload(mask):
    members=sum(1<<t for t in [1,2,3,4,5,7,9])
    return bool(mask&16384 and mask&members and not mask&~(16384|members|1))

def top_kind(k):return {0:1,1:2,4:4,7:5,8:6,9:7,11:8,13:12}.get(k,6 if 65<=k<128 else 3)

def resolved_seed(seed):return {1:0,2:1,4:4,5:7,6:8,7:9,8:11,12:13}.get(seed,2)

def concrete_kind(k):return k if k in {0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17} else 8 if 65<=k<128 else 3

class RepresentationShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='nano-representation-shapes-');cls.addClassCleanup(cls.workspace.cleanup);cls.directory=Path(cls.workspace.name)
        source=(ROOT/'src/nanoisa/nvm2c.c').read_bytes()
        markers=[b'#define NVM2C_VK_INT 0',b'static int variant_field(',
                 b'static int variant_field(',b'static uint8_t variant_field_for(',
                 b'static int variant_scalar_tags(',b'static int variant_scalar_kind(',
                 b'static void nvm2c_fail(',b'static int nvm2c_grow(',
                 b'typedef struct {\n    uint8_t kind;\n    int origin;',b'typedef struct {\n    uint8_t *parameters;',
                 b'static int shape_ok(',b'/* This opt-in shape is backed',
                 b'static int private_map_shape(',b'static int private_map_enabled(',
                 b'static int shape_record_return(',b'static int sim_push_slot(',
                 b'static uint8_t resolved_shape_kind(',b'static int sim_push(']
        assert all(source.count(m)==1 for m in set(markers));positions=[source.index(m) for m in markers]
        actual=b''.join(source[a:b] for a,b in zip(positions[::2],positions[1::2]));assert actual.count(b'static ')==16
        (cls.directory/'nvm2c_representation_shapes_actual.h').write_bytes(actual)
        (cls.directory/'reference-source.json').write_text(json.dumps({'source':'src/nanoisa/nvm2c.c','source_sha256':hashlib.sha256(source).hexdigest(),'verbatim_helper_sha256':hashlib.sha256(actual).hexdigest(),'functions':16,'logical_adapters':12,'minimal_owner_not_binary_layout':True,'product_source_edited':False,'markers':[m.decode() for m in markers]},indent=2)+'\n')
        cls.references=[]
        for name,flags in [('ordinary',['-O2']),('sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            binary=cls.directory/('reference-'+name)
            checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-Wall','-Wextra','-Werror','-D_POSIX_C_SOURCE=200809L',*flags,'-I',cls.directory,ROOT/'tests/nvm2c_representation_shapes_reference.c',ROOT/'src/nanoisa/isa.c',*native_link_flags(),'-o',binary],build=True)
            cls.references.append((name,[binary]))
        cls.fixture=ROOT/'tests/nvm2c_representation_shapes_driver.nano';cls.routes=build_routes(cls.fixture,cls.directory)
        assert len(cls.routes)==(5 if source_compiler_command() else 3)

    def compare(self, rows, label):
        paths = []
        manifest = {'qualification': False, 'cases': [], 'routes': []}
        for name, data in rows:
            path = self.directory / (label + '-' + name + '.bin')
            path.write_bytes(data)
            paths.append(path)
            manifest['cases'].append({'name': name, 'path': str(path),
                                      'sha256': hashlib.sha256(data).hexdigest()})
        baseline = None
        for name, command in [*self.references, *self.routes]:
            result = ''.join(checked([*command, *paths[i:i+16]], timeout=60) for i in range(0,len(paths),16))
            if baseline is None:
                baseline = result
            else:
                self.assertEqual(result, baseline, name)
            raw = result.encode()
            (self.directory / (label + '-' + name + '.reports')).write_bytes(raw)
            manifest['routes'].append({'name': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                                       'lines': len(result.splitlines())})
        (self.directory / (label + '-manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
        return baseline

    def test_complete_kind_and_inferred_field_domains(self):
        for op in [11,16]:
            cases=[(seed,k) for seed in range(15) for k in range(256)]
            programs=[program([operation(0,seed),operation(op,1,k),operation(19,1)]) for seed,k in cases]
            result=self.compare(grouped(programs),f'kind-{op}')
            events=[r.split('|') for r in result.splitlines() if r.startswith('E|')]
            self.assertEqual(len(events),len(cases)*3)
            for i,(seed,k) in enumerate(cases):
                admitted=k==2 or seed in [0,top_kind(k)]
                if op==16 and k in {0,1,9,11}:admitted=True;want=resolved_seed(seed)
                else:want=resolved_seed(seed) if k==2 else concrete_kind(k) if admitted else 2
                self.assertEqual(int(events[3*i+1][3]),int(admitted),(op,seed,k))
                self.assertEqual(int(events[3*i+1][4]),int(not admitted),(op,seed,k))
                self.assertEqual(int(events[3*i+2][3]),want,(op,seed,k))

    def test_complete_carrier_mask_and_extraction_domains(self):
        for op in [15,14]:
            programs=[]
            for mask in range(65536):
                steps=[operation(0,0),operation(15,1,mask),operation(19,1)] if op==15 else [operation(0,6),operation(1,1,0),operation(14,8,mask,1),operation(19,1)]
                programs.append(program(steps))
            result=self.compare(grouped(programs),f'masks-{op}')
            events=[r.split('|') for r in result.splitlines() if r.startswith('E|')]
            stride=3 if op==15 else 4
            self.assertEqual(len(events),65536*stride)
            if op==15:
                # I independently inspect the payload's root in each final snapshot.
                roots=[r.split('|') for r in result.splitlines() if r.startswith('N|3|')]
                self.assertEqual(len(roots),65536)
                for mask,row in enumerate(roots):self.assertEqual(int(row[3]),11 if payload(mask) and mask&128 else 10 if payload(mask) else 9,mask)
            for mask in range(65536):
                self.assertEqual(int(events[stride*mask+stride-2][3]),1 if op==15 else 3 if payload(mask) and mask&128 else 2,mask)
                self.assertEqual(int(events[stride*mask+stride-1][3]),8,mask)

    def test_every_single_record_field_pair(self):
        cases=[(a,b) for a in range(256) for b in range(256)]
        programs=[program([operation(0,0),operation(0,0),operation(18,1,2,fields=([a],[b]))]) for a,b in cases]
        result=self.compare(grouped(programs), 'record-pairs')
        events=[r.split('|') for r in result.splitlines() if r.startswith('E|') and r.split('|')[2]=='2']
        self.assertEqual(len(events),65536)
        self.assertTrue(all(int(r[3])==1 and int(r[4])==0 for r in events))

    def test_context_poison_disabled_and_source_diagnostics(self):
        kinds=[*range(18),65,80,96,127,255]
        rows=[program([operation(0,0),operation(op,1,k),operation(19,1)],flags=flags)
              for flags in range(32) for op in [11,16] for k in kinds]
        rows += [program([operation(0,0),operation(1,0,0),operation(op,1,kind),operation(8),operation(20,flags),operation(9,123),operation(19,1)],flags=flags)
                 for flags in range(32) for op in [11,16] for kind in [0,2,8,255]]
        rows += [program([operation(0,0),operation(0,0),operation(12,1,2),operation(13,1,4294967295),operation(12,1,2),operation(19,0)],flags=flags)
                 for flags in range(32)]
        rows += [program([operation(0,1),operation(10,1,2)],opcode=opcode,function=function,offset=offset)
                 for opcode in range(256) for function,offset in [(0,0),(4294967295,4294967295)]]
        result=self.compare(grouped(rows),'context')
        self.assertNotIn('ADAPTER_INPUT',result)
        # A fresh scalar flow can fail its graph without shape_ok copying a diagnostic.
        result=self.compare([('short-circuit',packet([program([operation(16,0,0),operation(8)])])),
                             ('disabled',packet([program([operation(1,0,0),operation(11,0,255),operation(19,0)],flags=8)]))], 'independent-context')
        lines=[r.split('|') for r in result.splitlines() if r.startswith('E|')]
        self.assertEqual(lines[0][3:7],['0','0','','1'])
        self.assertEqual(lines[0][7],'I cannot resolve an invalid shape ID')
        self.assertEqual(lines[1][3:5],['0','1'])
        self.assertEqual(lines[3][3:5],['1','1'])
        self.assertEqual(lines[4][3:5],['2','1'])

    def test_map_payload_and_record_sequence_boundaries(self):
        rows=[program([operation(0,seed),operation(17,1,tag),operation(19,1)],flags=flags)
              for seed in range(15) for tag in range(256) for flags in [1,5,0,4,13]]
        rows += [program([operation(0,0),operation(14,kind,mask,1),operation(8)],flags=flags)
                 for flags in range(32) for kind in [0,2,3,8,255] for mask in [0,6,16384,16386,16512,65535]]
        for flags in range(32):
            for width in [0,1,4,80,1024,65535]:
                fields=([2]*width,[2]*width)
                rows.append(program([operation(0,0),operation(0,0),operation(18,1,2,fields=fields),operation(4),operation(8)],flags=flags))
            rows.append(program([operation(0,0),operation(0,0),operation(18,1,2,fields=([0,1,8,96],[65,80,65,127])),operation(4),operation(8),operation(7),operation(20,flags),operation(9,0)],flags=flags))
        result=self.compare(grouped(rows),'map-record-contexts')
        self.assertNotIn('ADAPTER_INPUT',result)
        result=self.compare([('map-refusal',packet([program([operation(17,0,255)],flags=4)]))],'independent-map')
        self.assertTrue(result.startswith('E|0|0|0|0||0|\n'))

    def test_recursive_shared_graph_and_conversion_sequences(self):
        programs=[]
        for flags in range(32):
            steps=[operation(0,4),operation(0,4),operation(1,1,0),operation(2,3,1),operation(1,2,0),operation(2,4,2),operation(12,1,2),operation(19,1),operation(18,1,2,fields=([2],[2])),operation(4),operation(8)]
            programs.append(program(steps,flags=flags))
            programs.append(program([operation(0,0),operation(16,1,0),operation(16,1,1),operation(4),operation(8),operation(7),operation(20,flags),operation(9,0)],flags=flags))
        for depth in [1,4,80,1024,10000]:
            steps=[operation(0,4)]
            for i in range(depth):steps.append(operation(13,i+1,0))
            steps += [operation(10,depth+1,1),operation(19,1),operation(8)]
            programs.append(program(steps,mode=1))
        result=self.compare(grouped(programs,limit=8),'sequences')
        self.assertNotIn('ADAPTER_INPUT',result)

    def test_malformed_and_prior_packet_boundaries(self):
        sample=packet([program([operation(9),operation(18,1,1,fields=([8],[65]))])])
        cases=[(f'truncated-{n}',sample[:n]) for n in range(len(sample))]
        cases += [('late-op',packet([program([operation(9),operation(21)])])),('flags',packet([program([],flags=32)])),('mode',packet([program([],mode=2)])),
                  ('mask',packet([program([operation(15,1,65536)])])),('shape-kind',packet([program([operation(10,1,15)])])),('kind',packet([program([operation(11,1,256)])])),
                  ('payload',packet([program([operation(14,256,0,0)])])),('reset-flags',packet([program([operation(20,32)])])),('size',b'\0'*1048577),('huge-count',struct.pack('<I',0xffffffff)),('trailing',sample+b'\0')]
        self.assertEqual(self.compare(cases,'malformed'),'INVALID\n'*len(cases))
        valid=packet([program([operation(9)])]);neighbors=[('before',valid),('refused',sample+b'\0'),('after',valid),('empty',packet([]))]
        output=self.compare(neighbors,'neighbors')
        one='E|0|0|1|0||1|\nZ|0|1|0|0||1|\nN|1|1|0|0|0||0|\nD|0|0|0|0\nI|1|0\n'
        self.assertEqual(output,one+'INVALID\n'+one)

    def test_imported_shadows(self):
        binary = self.directory / 'imported-native'
        checked([ROOT / 'bin/nanoc_c', self.fixture, '--test-imports', '-o', binary], build=True)
        self.assertEqual(checked([binary]), '')
        for name, command in [('c', [ROOT / 'bin/nano_virt']), ('nanolang', source_compiler_command())]:
            if not command:
                continue
            module = self.directory / ('imported-' + name + '.nvm')
            checked([*command, self.fixture, '--test-imports', '--emit-nvm', '-o', module], build=True)
            checked([ROOT / 'bin/nano_vm', '--verify-only', module])
            self.assertEqual(checked([ROOT / 'bin/nano_vm', module, '--']), '')


if __name__=='__main__':unittest.main()
