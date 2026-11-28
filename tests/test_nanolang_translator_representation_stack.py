"""I compare representation stack helpers with their verbatim C bodies."""
from pathlib import Path
import hashlib,json,os,shlex,struct,tempfile,unittest
from tests.nanolang_translator_routes import build_routes,checked,source_compiler_command
from tests.native_process import native_link_flags
ROOT=Path(__file__).resolve().parents[1]
LIMIT=4194304

def operation(code,kind=2,origin=-1,shape=0,tags=0,payload=0,fields=None,parameter=0):
    return struct.pack('<BBBBiIHHI',code,kind,fields is not None,payload,origin,shape,tags,0,parameter)+(b'' if fields is None else bytes(fields))

def program(steps,*,flags=0,opcode=0,mode=0,function=17,offset=29,argument=7,capacity=2,current=0,defaults=(2,2),locals=(2,8,9)):
    return struct.pack('<BBBBIIIQIHHI',flags,opcode,mode,0,function,offset,argument,capacity,current,len(defaults),len(locals),len(steps))+bytes(defaults)+bytes(locals)+b''.join(steps)

def packet(programs):return struct.pack('<I',len(programs))+b''.join(programs)

def grouped(programs,prefix,maximum=128):
    rows=[];batch=[];size=4
    for data in programs:
        if batch and (len(batch)>=maximum or size+len(data)>LIMIT):
            rows.append((f'{prefix}-{len(rows):04d}',packet(batch)));batch=[];size=4
        assert len(data)+4<=LIMIT
        batch.append(data);size+=len(data)
    if batch:rows.append((f'{prefix}-{len(rows):04d}',packet(batch)))
    return rows

def actual_helpers(source):
    spans=[
        (b'#define NVM2C_VK_INT 0',b'static int variant_field('),
        (b'static int variant_field(',b'static uint8_t variant_field_for('),
        (b'static int variant_scalar_tags(',b'static int variant_scalar_kind('),
        (b'static int boolean_result(',b'typedef struct Nvm2cFieldBlock'),
        (b'#define NVM2C_SCALAR_UNKNOWN',b'/* A boxed carrier is not proof'),
        (b'static uint16_t scalar_kind_tags(',b'static int optional_scalar_tags('),
        (b'static void nvm2c_fail(',b'static int nvm2c_grow('),
        (b'typedef struct {\n    uint8_t kind;\n    int origin;',b'typedef struct {\n    uint8_t *parameters;'),
        (b'static int shape_ok(',b'static int shape_equal('),
        (b'static NvmShapeId shape_child(',b'/* I extract the admitted integer-array member'),
        (b'static int shape_carrier_box(',b'/* This opt-in shape is backed'),
        (b'static int sim_push_slot(',b'/* Read only resolved representation facts.'),
        (b'static int sim_push(',b'static const uint8_t *fn_rec_k_const('),
    ]
    for a,b in spans:assert source.count(a)==source.count(b)==1,(a,b)
    result=b''.join(source[source.index(a):source.index(b)] for a,b in spans)
    assert result.count(b'static ')==20
    return result,[[a.decode(),b.decode()] for a,b in spans]

class RepresentationStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not source_compiler_command():raise RuntimeError('I require an explicitly selected genuine qualified NanoLang compiler.')
        cls.workspace=tempfile.TemporaryDirectory(prefix='nano-representation-stack-');cls.addClassCleanup(cls.workspace.cleanup);cls.directory=Path(cls.workspace.name)
        source=(ROOT/'src/nanoisa/nvm2c.c').read_bytes();actual,spans=actual_helpers(source)
        (cls.directory/'nvm2c_representation_stack_actual.h').write_bytes(actual)
        (cls.directory/'reference-source.json').write_text(json.dumps({'source':'src/nanoisa/nvm2c.c','source_sha256':hashlib.sha256(source).hexdigest(),'helper_sha256':hashlib.sha256(actual).hexdigest(),'verbatim_helpers':20,'minimal_owner_not_binary_layout':True,'spans':spans,'product_source_changed':False},indent=2)+'\n')
        cls.references=[]
        for name,flags in [('ordinary',['-O2']),('sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])]:
            binary=cls.directory/('reference-'+name)
            checked([*shlex.split(os.environ.get('CC','cc')),'-std=c11','-Wall','-Wextra','-Werror','-D_POSIX_C_SOURCE=200809L',*flags,'-I',cls.directory,ROOT/'tests/nvm2c_representation_stack_reference.c',ROOT/'src/nanoisa/isa.c',*native_link_flags(),'-o',binary],build=True)
            cls.references.append((name,[binary]))
        cls.routes=build_routes(ROOT/'tests/nvm2c_representation_stack_driver.nano',cls.directory)
        assert len(cls.routes)==5

    def compare(self,rows,label):
        paths=[];manifest={'qualification':False,'whole_release':False,'cases':[],'routes':[]}
        for name,data in rows:
            path=self.directory/(label+'-'+name+'.bin');path.write_bytes(data);paths.append(path)
            manifest['cases'].append({'name':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
        baseline=None
        for name,command in [*self.references,*self.routes]:
            text=''.join(checked([*command,*paths[i:i+8]],timeout=60) for i in range(0,len(paths),8))
            if baseline is None:baseline=text
            else:self.assertEqual(text,baseline,name)
            raw=text.encode();(self.directory/(label+'-'+name+'.reports')).write_bytes(raw)
            manifest['routes'].append({'name':name,'sha256':hashlib.sha256(raw).hexdigest(),'lines':len(text.splitlines())})
        (self.directory/(label+'-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
        return baseline

    def test_named_current_shape_refusals_and_branch_storage(self):
        rows=[
            ('boolean',packet([program([operation(1,0,7)],opcode=3,current=99)])),
            ('first-error',packet([program([operation(1,0),operation(1,1),operation(2,11,123)],flags=1,capacity=1)])),
            ('failed-current',packet([program([operation(1,0)],flags=25)])),
            ('shared-locals',packet([program([operation(1,0,0),operation(6),operation(2),operation(1,1,2),operation(4,11,0),operation(7),operation(2)])])),
            ('unknown-string',packet([program([operation(0,2,0),operation(2),operation(3,2,0,shape=1),operation(9)],flags=1,opcode=252)])),
            ('untracked-failure',packet([program([operation(1,0),operation(2)],flags=24)])),
        ]
        text=self.compare(rows,'named')
        self.assertIn('S|0|9|7|1|0|16|0|2,2\n',text)
        self.assertIn('E|0|0|0|1|prior|0|1|2|\n',text)
        self.assertIn('function 7: operand stack overflow',text)
        self.assertIn('V|-1|0|0|1|0|2|0|2,2\n',text)
        self.assertIn('L|0,8,9\n',text)
        self.assertIn('L|1,8,9\n',text)
        self.assertIn('E|0|0|1|1|prior|1|0|0|\n',text)

    def test_all_opcode_kind_masks_and_projection_sequences(self):
        programs=[program([operation(1,0)],opcode=op) for op in range(256)]
        programs += [program([operation(0,kind)],flags=flags) for flags in [0,1,3,9,25] for kind in range(256)]
        programs += [program([operation(0,8,tags=mask)],flags=1,mode=1,defaults=(),locals=()) for mask in range(65536)]
        for opcode in [0,7,16,244,252]:
            for kind in [0,1,2,4,7,8,9,11,13,65,96,255]:
                for flags in [0,1,3,17,25]:
                    programs.append(program([operation(0,kind,0,fields=(0,255),payload=255),operation(2),operation(5,parameter=0),operation(8,parameter=opcode),operation(1,0),operation(9)],flags=flags,opcode=opcode))
        self.compare(grouped(programs,'matrix'),'matrix')

    def test_typed_consumers_and_defined_origin_boundaries(self):
        programs=[]
        for operation_code in [3,4]:
            for kind in range(256):
                for origin in [-2147483648,-1,0,1,2,3,65535,65539,2147483647]:
                    for flags in [0,1,24]:
                        programs.append(program([operation(operation_code,kind,origin)],flags=flags))
        programs += [program([operation(3,2,0,shape=1),operation(4,2,0,shape=1)],flags=1),
                     program([operation(4,1,0),operation(3,8,1),operation(4,8,2)],flags=0),
                     program([operation(4,0,65536)],locals=()),
                     program([operation(3,1,65536)],locals=(8,)),
                     program([operation(4,11,0)],locals=bytes([2])*65535)]
        text=self.compare(grouped(programs,'consumers'),'consumers')
        self.assertIn('I require integer operands for typed I64 operations',text)
        self.assertIn('INVALID\n',text)

    def test_exact_stack_height_storage_reuse_and_transport(self):
        programs=[program([operation(1,0),operation(2,11,123)],capacity=cap,current=current) for cap in [0,1,2,65535,4294967295,4294967296] for current in [0,1,4294967295]]
        programs += [program([operation(1,kind),operation(2),operation(1,kind)],defaults=bytes(range(256)),locals=(),opcode=opcode) for kind in [0,1,2,8,255] for opcode in [0,6,108,143]]
        push=operation(1,0);pop=operation(2,11,123)
        # I fill a configured 65535-cell stack, refuse one extra push, drain it,
        # and preserve the first error through a final underflow. No report truncates.
        deep=program([push]*65536+[pop]*65536,capacity=65535,mode=1,defaults=(),locals=())
        rows=grouped(programs,'capacity')+[('uint16-frontier',packet([deep]))]
        valid=packet([program([operation(1,0)])]);bad=[b'',b'\0'*3,valid+b'\0',b'\xff'*4,b'\0'*(LIMIT+1)]
        for index in [4,6,7,24,25,26,27,45,47,59,60]:
            if index<len(valid):
                data=bytearray(valid);data[index]=255;bad.append(bytes(data))
        text=self.compare(rows,'bounds')
        self.assertIn('|65535|0|0|\n',text)
        self.assertIn('function 7: operand stack overflow',text)
        invalid=self.compare([(f'malformed-{i}',data) for i,data in enumerate(bad)],'framing')
        self.assertEqual(invalid,'INVALID\n'*len(bad))
