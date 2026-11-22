"""I compare complete local state traces with actual C transitions and queries."""
from pathlib import Path
import hashlib
import json
import os
import random
import struct
import tempfile
import unittest
from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_ownership import (
    desc, record, function, image, extension, effect_data, corpus as declarations,
)

ROOT = Path(__file__).resolve().parents[1]
NO = 0xffffffff


def command(op, source=0, a=0, b=0, c=0, d=0, fields=()):
    return (op, source, a, b, c, d, len(fields), *fields)


def program(model, commands):
    return struct.pack('<II', len(model), len(commands))+model+b''.join(
        struct.pack('<'+'I'*len(cmd), *cmd) for cmd in commands)


def basic_model(scalar=1, borrowed_mode=2, result=None, flags=(3, 3, 3)):
    rows = [(0, NO, [(1, NO, NO)]),
            (0, NO, [(8, 0, NO), (8, 0, NO)]), (0, NO, [(1, NO, NO)])]
    locals_ = [desc(scalar), desc(1), *[desc(8, 0)]*3, *[desc(8, 1)]*2, desc(8, 2)]
    return image(record(1, flags=flags, functions=[(8, 0, desc() if result is None else result, locals_),
        (1, 1, desc(), [desc(8, 0, borrowed_mode)])]), rows=rows,
        functions=[function(8, results=int(result is not None), result_tag=0 if result is None else result[0]),
                   function(1, 1, params=(8,))])


def union_model(owned=True):
    strings = [b'Handle', b'Choice', b'Outer', b'Some', b'Empty', b'Pair', b'value']
    rows = [(0, 0, [(1, NO, 6)]), (2, 1, [(8, 0, 6)]*3), (2, 2, [(10, 1, 6)])]
    unions = struct.pack('<I', 2)
    for layout, variants in [(1, [(3, 0, 1), (4, 1, 0), (5, 1, 2)]), (2, [(3, 0, 1)])]:
        unions += struct.pack('<IHH', layout, len(variants), 0)
        unions += b''.join(struct.pack('<IHH', *v) for v in variants)
    locals_ = [desc(1), desc(8, 0), desc(8, 0), desc(10, 1), desc(10, 1), desc(10, 2), desc(10, 2)]
    return image(record(3, flags=[3 if owned else 1]*3,
        functions=[(7, 0, desc(), locals_)], extensions=[extension(1, unions)]),
        rows=rows, strings=strings, functions=[function(7)])


def array_model():
    types = [desc(1), desc(7, 0), desc(8, 0), desc(7, 2), desc(1), desc(8, 0)]
    locals_ = [desc(7, i) for i in range(len(types))]+[desc(1), desc(8, 0), desc(5)]
    return image(record(3, flags=[3], functions=[(len(locals_), 0, desc(), locals_)],
        extensions=[extension(4, effect_data(2, types=types), 2)]),
        rows=[(0, NO, [(1, NO, NO)])], strings=[b'op'], functions=[function(len(locals_))])


def handler_model():
    locals_ = [desc(1), desc(1), desc(8, 0), desc(1), desc(8, 0)]
    effects = effect_data(operations=[(0, desc(1), [desc(1), desc(8, 0)])])
    return image(record(3, flags=[3], functions=[(5, 0, desc(), locals_)],
        extensions=[extension(4, effects)]), rows=[(0, NO, [(1, NO, NO)])],
        strings=[b'op'], functions=[function(5)])


def signature_model(result=0, *, depth=1, fields=1, leaf=1, parameters=(), flags=3):
    rows = [(0, NO, [(leaf, NO, NO)]*fields)]
    rows += [(0, NO, [(8, i-1, NO)]) for i in range(1, depth)]
    result_slot = desc(result, depth-1 if result == 8 else NO)
    values = [desc(tag, 0 if tag == 8 else NO, mode) for tag, mode in parameters]
    ownership = record(1, flags=[flags]*depth,
                       functions=[(len(values), len(values), result_slot, values)])
    return image(ownership, rows=rows, functions=[function(len(values), len(values),
        results=int(result != 0), result_tag=result, params=tuple(tag for tag, _ in parameters))])


def corpus(include_declarations=True):
    cases = []

    def add(name, model, commands, expectations=None):
        cases.append((name, program(model, commands), expectations or {}))

    create = command(0, b=8)
    make_pair = [create, command(2, a=0), command(9, a=2, fields=(0,)),
                 command(9, a=3, fields=(0,)), command(9, a=5, fields=(2, 3))]
    consume = [command(10, a=5, fields=(2, 3)), command(10, a=2, fields=(0,)),
               command(10, a=3, fields=(0,)), command(19, a=65535)]
    add('original-pack-unpack', basic_model(), make_pair+consume,
        {i: True for i in range(len(make_pair+consume))})
    borrow = make_pair+[command(13), command(15, a=0, b=5, c=1, fields=(0,)),
        command(15, a=1, b=5, c=2, fields=(1,)), command(18, a=0, b=0),
        command(18, a=0, b=0, c=1), command(18, a=1, b=0, c=1),
        command(17, a=5, b=0, fields=(0, 0)), command(17, a=5, b=1, fields=(0, 0)),
        command(17, a=5, b=0, fields=(1, 0)), command(14), *consume]
    add('original-disjoint-borrow-access', basic_model(), borrow,
        {0: True, 6: True, 7: True, 8: True, 9: False, 10: True, 11: True, 12: False, 13: False, 14: True, len(borrow)-1: True})
    isolated = make_pair+[command(13), command(15, a=0, b=5, c=2, fields=(0,)),
        command(1, a=1), command(17, source=1, a=5, fields=(0, 0)),
        command(14), command(18, source=1, a=0, b=0), command(41, a=1),
        command(14, source=1), command(41, a=1)]
    add('borrow-path-clone-isolation', basic_model(), isolated,
        {6: True, 7: True, 8: False, 9: True, 10: True, 11: False, 12: True, 13: True})
    for mode in (1, 2):
        commands = [command(0, a=1, b=4), command(45, a=0), command(46),
            command(18, a=0, b=0, c=1), command(13), command(16, a=1, b=0, c=1),
            command(18, a=0, b=0, c=1), command(18, a=1, b=0), command(14), command(19, a=65535)]
        add(f'original-reference-parameter-{mode}', basic_model(borrowed_mode=mode), commands,
            {0: True, 1: True, 2: True, 3: mode == 2, 4: True, 5: True, 6: False, 7: True, 8: True, 9: True})
    for scalar in (1, 2, 3, 4):
        commands = [create, command(1, a=1), command(2, a=0), command(41, a=1),
            command(42, a=1), command(20, a=0), command(41, a=1),
            command(0, source=2, b=8), command(41, a=2), command(42, a=2)]
        add(f'original-scalar-meet-{scalar}', basic_model(scalar=scalar), commands,
            {0: True, 1: True, 2: True, 3: False, 4: True, 5: False, 6: True, 7: True, 8: False, 9: False})
    union_trace = [create, command(39), command(2, a=0), command(9, a=1, fields=(0,)),
        command(6, a=3, b=1, c=0), command(11, a=3, b=1), command(11, a=3, b=0, fields=(1,)),
        command(28, a=3), command(8, a=3, b=4), command(28, a=4),
        command(12, a=4, b=1), command(12, a=4, b=0, fields=(1,)),
        command(10, a=1, fields=(0,)), command(19, a=65535),
        command(11, a=3, b=1), command(1, a=1), command(7, a=3, b=0),
        command(12, a=3, b=1), command(19, a=65535), command(19, source=1, a=65535)]
    add('original-union-transfers', union_model(), union_trace,
        {0: True, 1: True, 2: True, 3: True, 4: False, 5: True, 6: False, 7: True, 8: True, 9: True,
         10: True, 11: False, 12: True, 13: True, 14: True, 15: True, 16: False, 17: True, 18: True, 19: False})
    # I retain exact resource obligations while forgetting differing selected arms.
    add('union-selected-meet', union_model(), [create, command(2, a=0), command(1, a=1),
        command(11, a=3, b=1), command(9, source=1, a=1, fields=(0,)),
        command(11, source=1, a=3, b=0, fields=(1,)), command(42, a=1), command(28, a=3),
        command(19, a=65535), command(12, a=3, b=1)],
        {0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: False, 8: False, 9: False})
    for owned in (False, True):
        cmds = [create]
        for local in range(7):
            cmds += [command(20, a=local), command(21, a=local), command(35, a=local, b=8, c=1), command(28, a=local)]
        for variant in (0, 1, 2, 3, 65535):
            cmds += [command(6, a=3, b=1, c=variant), command(7, a=3, b=variant), command(28, a=3)]
        add(f'union-queries-{owned}', union_model(owned), cmds)
    cmds = [create, command(40)]
    for local in range(9):
        cmds += [command(21, a=local), command(5, a=local, b=local), command(30, a=7, b=local),
                 command(36, a=7, b=local), command(31, a=local)]
    for tag in range(16): cmds += [command(38, a=tag), command(37, a=tag, b=NO)]
    for a in range(6):
        for b in range(6): cmds += [command(29, a=7, b=a, c=7, d=b)]
    cmds += [command(32, a=2, b=7, c=2), command(31, a=2), command(32, a=3, b=7, c=3), command(31, a=3)]
    add('typed-array-queries-transfers', array_model(), cmds, {0: True, 1: True, len(cmds)-1: True})
    cmds = [create, command(2, a=0), command(9, a=2, fields=(0,)), command(43, a=1, b=0, c=3),
        command(17, source=1, a=2), command(9, source=1, a=2, fields=(0,)),
        command(20, source=1, a=0), command(27, source=1, a=1, b=NO),
        command(10, source=1, a=4, fields=(1,)), command(27, source=1, a=1, b=NO),
        command(27, source=1, a=4, b=NO), command(20, a=2)]
    add('handler-activation-resume', handler_model(), cmds,
        {0: True, 1: True, 2: True, 3: True, 4: False, 5: False, 6: True, 7: False, 8: True, 9: True, 10: False, 11: True})
    for tag in (0, 1, 2, 3, 4, 5, 8):
        add(f'value-result-tag-{tag}', signature_model(tag), [create, command(48)],
            {0: True, 1: tag not in (3, 5)})
    for depth in (1, 2, 31, 32, 33):
        for fields in (0, 1, 256, 257):
            add(f'value-result-tree-{depth}-{fields}', signature_model(8, depth=depth, fields=fields),
                [create, command(48)], {0: True, 1: depth <= 32 and (depth == 1 or fields <= 256)})
    add('value-result-float-leaf', signature_model(8, leaf=3), [create, command(48)], {0: True, 1: False})
    for count in (0, 1, 2, 8, 9):
        for tag in (1, 2, 3, 4, 5, 8):
            for mode in ((0, 1, 2) if tag == 8 else (0,)):
                model = signature_model(parameters=[(tag, mode)]*count)
                cmds = [command(0, b=max(8, count))]
                for capacity in (max(0, count-1), count, count+1):
                    cmds += [command(49, a=capacity, b=1), command(50, a=capacity, b=1)]
                cmds += [command(45, a=0), command(46), command(47), command(49, a=count, b=0)]
                valid = count == 0 or (count <= 8 and tag != 3 and mode == 0)
                wants = {0: True, 3: valid, 4: valid and count > 0 and tag == 8, len(cmds)-1: False}
                add(f'value-parameters-{count}-{tag}-{mode}', model, cmds, wants)
    for mode in (1, 2):
        for caller_module in (0, 1):
            cmds = make_pair+[command(13), command(15, a=0, b=5, c=mode, fields=(0,)),
                command(0, source=1, a=1, b=4, c=caller_module), command(44, source=1, a=0),
                command(20, source=1, a=0), command(18, source=1, a=0, b=0, c=1),
                command(13, source=1), command(16, source=1, a=1, b=0, c=1), command(14, source=1), command(20, a=5)]
            add(f'caller-binding-{mode}-{caller_module}', basic_model(), cmds,
                {0: True, 6: True, 7: True, 8: mode == 2 and caller_module == 0, 14: mode == 1})
    # Every operation is observed on a null state and a fresh valid state. I
    # preserve the operation's native widths instead of narrowing large inputs.
    for op in range(2, 52):
        if op in (41, 42, 43, 44): continue
        for local in (0, 1, 2, 5, 7, 8, 65535):
            args = dict(a=local, b=0, c=0, d=0)
            add(f'operation-{op}-{local}', basic_model(), [command(op, **args), create, command(op, **args)])
    for kind in ('ready', 'borrowed'):
        prefix = make_pair + ([command(13), command(15, a=0, b=5, c=2, fields=(0,))] if kind == 'borrowed' else [])
        for local in range(9):
            for target in range(9):
                add(f'move-{kind}-{local}-{target}', basic_model(), prefix+[command(8, a=local, b=target)])
        for mode in (0, 1, 2, 3, NO):
            for fields in ((), (0,), (1,), (0, 0), (2,), (65535,)):
                add(f'borrow-{kind}-{mode}-{fields}', basic_model(), prefix+[command(15, a=1, b=5, c=mode, fields=fields)])
    rng = random.Random(1142012)
    for number in range(128):
        commands = [create]
        for _ in range(24):
            op = rng.choice((2, 3, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 31, 32, 35))
            fields = tuple(rng.randrange(9) for _ in range(rng.randrange(3))) if op in (9, 10, 15, 17) else ()
            commands.append(command(op, a=rng.randrange(9), b=rng.randrange(9), c=rng.randrange(3), fields=fields))
        add(f'random-state-trace-{number}', basic_model(), commands)
    if include_declarations:
        for name, data, _ in declarations():
            add('declaration-create-'+name, data, [create, command(39), command(40), command(48), command(49, a=8, b=1)])
    model = basic_model()
    valid = program(model, [create])
    for number, data in enumerate((b'', valid[:7], valid[:-1], valid+b'\0',
            program(model, [command(52)]), program(model, [command(0, source=16)]),
            program(model, [command(0, b=65537)]), program(model, [command(0, c=2)]),
            program(model, [command(1, a=16)]), program(model, [command(2, a=65536)]),
            program(model, [command(24, a=256)]), program(model, [command(29, c=256)]),
            program(model, [command(34, c=65536)]), program(model, [command(15, fields=(65536,))]))):
        cases.append((f'harness-refusal-{number}', data, {'carrier': 'D'}))
    return cases


class AffineState(unittest.TestCase):
    def test_actual_transition_traces(self):
        (ROOT/'.test_output').mkdir(exist_ok=True)
        requested = os.environ.get('NANOLANG_AFFINE_STATE_PRODUCTS')
        if requested:
            work = Path(requested); work.mkdir(parents=True, exist_ok=False)
        else:
            work = Path(tempfile.mkdtemp(prefix='affine-state-', dir=ROOT/'.test_output'))
        print(f'I retain affine-state products at {work}', flush=True)
        cases = corpus(); paths = []
        record_ = dict(cases=len(cases), inputs=[], routes=[], independent_expectations=sum(len(w) for _, _, w in cases))
        for i, (name, data, want) in enumerate(cases):
            p = work/f'case-{i}.bin'; p.write_bytes(data); paths.append(p)
            record_['inputs'].append(dict(name=name, file=p.name, expected=want, sha256=hashlib.sha256(data).hexdigest()))
        def save(): (work/'manifest.json').write_text(json.dumps(record_, indent=2)+'\n')
        save()
        def execute(command_, name):
            reports = []
            for offset in range(0, len(paths), 16):
                reports += checked([*command_, *paths[offset:offset+16]], timeout=600).splitlines()
            self.assertEqual(len(reports), len(cases), name)
            (work/(name+'.reports')).write_text('\n'.join(reports)+'\n')
            return reports
        oracle = Path(os.environ.get('NANOLANG_AFFINE_STATE_REFERENCE', ROOT/'obj/nvm2c_affine_state_reference'))
        expected = execute([oracle], 'actual-c')
        for (name, _, want), got in zip(cases, expected):
            steps = got.split('|')
            if 'carrier' in want:
                with self.subTest(expectation=name): self.assertEqual(got, want['carrier'])
            else:
                for index, accepted in want.items():
                    with self.subTest(expectation=name, step=index): self.assertEqual(steps[index][0], str(int(accepted)))
        for name, cmd in build_routes(ROOT/'tests/nvm2c_affine_state_driver.nano', work):
            actual = execute(cmd, name)
            for (case, _, _), want, got in zip(cases, expected, actual):
                with self.subTest(route=name, case=case): self.assertEqual(got, want)
            record_['routes'].append(dict(name=name, reports=len(actual), command=list(map(str, cmd))))
            save()
        print(f'I match {len(cases)} actual affine state traces through {len(record_["routes"])} routes.', flush=True)
