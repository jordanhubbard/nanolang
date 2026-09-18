"""I recover a bounded typed region tree; both emitters consume that tree."""
from dataclasses import dataclass, field


class Refusal(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Refusal('I ' + message)


INT, BOOL = 1, 4
COMPARE = {'I64_EQ': '==', 'I64_NE': '!=', 'I64_LT_S': '<',
           'I64_LE_S': '<=', 'I64_GT_S': '>', 'I64_GE_S': '>='}
UNSIGNED_COMPARE = {'I64_LT_U': 'lt_u', 'I64_LE_U': 'le_u',
                    'I64_GT_U': 'gt_u', 'I64_GE_U': 'ge_u'}
GENERIC_COMPARE = {'EQ': '==', 'NE': '!=', 'LT': '<', 'LE': '<=', 'GT': '>', 'GE': '>='}
BRANCH = {'JMP_TRUE', 'JMP_FALSE'}
ARITHMETIC = {'ADD': 'add', 'SUB': 'sub', 'MUL': 'mul', 'DIV': 'div', 'MOD': 'rem', 'NEG': 'neg',
              'I64_ADD': 'add', 'I64_SUB': 'sub', 'I64_NEG': 'neg', 'I64_MUL': 'mul',
              'I64_DIV_S': 'div', 'I64_REM_S': 'rem',
              'I64_DIV_U': 'div_u', 'I64_REM_U': 'rem_u',
              'I64_SHL': 'shl', 'I64_SHR_S': 'shr_s', 'I64_SHR_U': 'shr_u',
              'I64_AND': 'band', 'I64_OR': 'bor', 'I64_XOR': 'bxor', 'I64_INVERT': 'invert'}
SIMPLE = {'NOP', 'PUSH_I64', 'PUSH_BOOL', 'LOAD_LOCAL', 'STORE_LOCAL',
          'DUP', 'POP', 'SWAP', 'ROT3', 'PICK', 'ROLL', 'BOOL_AND', 'BOOL_OR', 'BOOL_NOT', 'CALL',
          'CAST_INT', 'CAST_BOOL', 'AND', 'OR', 'NOT', 'I64_MUL_WIDE_S', 'I64_MUL_WIDE_U'} | set(COMPARE) | set(ARITHMETIC) | set(UNSIGNED_COMPARE) | set(GENERIC_COMPARE)


@dataclass(frozen=True)
class Expr:
    tag: int
    kind: str
    value: object
    args: tuple = ()
    size: int = field(init=False)
    depth: int = field(init=False)

    def __post_init__(self):
        size = 1 + sum(arg.size for arg in self.args)
        depth = 1 + max((arg.depth for arg in self.args), default=0)
        require(size <= 4096 and depth <= 128, 'limit expanded scalar expressions to 4096 nodes and depth 128')
        object.__setattr__(self, 'size', size)
        object.__setattr__(self, 'depth', depth)


@dataclass
class Function:
    name: str
    params: list
    result: int
    locals: dict
    names: list
    body: list


class Analyze:
    def __init__(self, module, index):
        self.module = module
        self.fn = module['functions'][index]
        self.index = index
        self.code = self.fn['code']
        self.positions = {i['pc']: n for n, i in enumerate(self.code)}
        self.positions[self.fn['size']] = len(self.code)
        self.types = dict(enumerate(self.fn['params']))
        self.seen = set()
        self.calls = set()
        self.backedges = {}
        for n, i in enumerate(self.code):
            require(i['op'] in SIMPLE | BRANCH | {'JMP', 'RET'},
                    'do not reconstruct opcode ' + i['op'])
            if i['op'] in BRANCH | {'JMP'}:
                require(i['target'] in self.positions, 'require exact branch boundaries')
                target = self.positions[i['target']]
                if target <= n:
                    require(i['op'] == 'JMP', 'require pretest loops with one unconditional backedge')
                    require(target not in self.backedges, 'require one backedge per loop header')
                    self.backedges[target] = n

    def touch(self, index):
        require(index not in self.seen, 'require disjoint structured regions')
        self.seen.add(index)

    def pop(self, stack, tag=None):
        require(bool(stack), 'require a populated scalar operand stack')
        value = stack.pop()
        require(tag is None or value.tag == tag, 'require exact scalar operand types')
        return value

    @staticmethod
    def truth(value):
        require(value.tag in (INT, BOOL), 'require exact int/bool truth operands')
        return value if value.tag == BOOL else Expr(BOOL, 'binary', '!=',
                                                    (value, Expr(INT, 'constant', 0)))

    def simple(self, index, stack, initialized, statements, pure=False):
        self.touch(index)
        ins = self.code[index]
        op, arg = ins['op'], ins['arg']
        expr = None
        if op == 'NOP':
            return
        if op == 'PUSH_I64':
            expr = Expr(INT, 'constant', arg)
        elif op == 'PUSH_BOOL':
            require(arg in (0, 1), 'require canonical boolean constants')
            expr = Expr(BOOL, 'constant', bool(arg))
        elif op == 'LOAD_LOCAL':
            require(arg in initialized, 'require a definitely initialized local')
            expr = Expr(self.types[arg], 'local', arg)
        elif op == 'STORE_LOCAL':
            require(not pure, 'require side-effect-free loop conditions')
            value = self.pop(stack)
            require(0 <= arg < self.fn['locals'], 'require a declared local slot')
            require(arg not in self.types or self.types[arg] == value.tag,
                    'require one scalar type per local slot')
            self.types[arg] = value.tag
            initialized.add(arg)
            statements.append(('store', arg, value))
            return
        elif op == 'DUP':
            value = self.pop(stack)
            stack.extend((value, value))
            return
        elif op == 'POP':
            self.pop(stack)
            return
        elif op in ('PICK', 'ROLL'):
            require(0 <= arg < len(stack), 'require an indexed operand within the current scalar stack')
            index = len(stack) - 1 - arg
            value = stack[index] if op == 'PICK' else stack.pop(index)
            stack.append(value)
            return
        elif op == 'SWAP':
            right, left = self.pop(stack), self.pop(stack)
            stack.extend((right, left))
            return
        elif op == 'ROT3':
            top, middle, bottom = self.pop(stack), self.pop(stack), self.pop(stack)
            stack.extend((top, bottom, middle))
            return
        elif op in ('I64_MUL_WIDE_S', 'I64_MUL_WIDE_U'):
            right, left = self.pop(stack, INT), self.pop(stack, INT)
            for word, helper in (('low', 'mul'), ('high', 'mul_high_s' if op.endswith('_S') else 'mul_high_u')):
                result = Expr(INT, 'arithmetic', helper, (left, right))
                if pure:
                    stack.append(result)
                else:
                    temporary = Expr(INT, 'temporary', f'{ins["pc"]}_{word}')
                    statements.append(('let', temporary, result))
                    stack.append(temporary)
            return
        elif op in ARITHMETIC:
            right = self.pop(stack, INT)
            args = (right,) if op in ('NEG', 'I64_NEG', 'I64_INVERT') else (self.pop(stack, INT), right)
            expr = Expr(INT, 'arithmetic', ARITHMETIC[op], args)
        elif op in UNSIGNED_COMPARE:
            right, left = self.pop(stack, INT), self.pop(stack, INT)
            expr = Expr(BOOL, 'unsigned_compare', UNSIGNED_COMPARE[op], (left, right))
        elif op in GENERIC_COMPARE:
            right, left = self.pop(stack), self.pop(stack)
            require(left.tag in (INT, BOOL) and right.tag in (INT, BOOL),
                    'require exact int/bool generic comparison operands')
            if left.tag != right.tag:
                # Ordinary operand evaluations already have immutable statements.
                result = {'EQ': False, 'NE': True, 'LT': left.tag < right.tag,
                          'LE': left.tag < right.tag, 'GT': left.tag > right.tag,
                          'GE': left.tag > right.tag}[op]
                expr = Expr(BOOL, 'constant', result)
            else:
                if left.tag == BOOL and op not in ('EQ', 'NE'):
                    left = Expr(INT, 'bool_int', None, (left,))
                    right = Expr(INT, 'bool_int', None, (right,))
                expr = Expr(BOOL, 'binary', GENERIC_COMPARE[op], (left, right))
        elif op in COMPARE:
            right, left = self.pop(stack, INT), self.pop(stack, INT)
            expr = Expr(BOOL, 'binary', COMPARE[op], (left, right))
        elif op in ('CAST_BOOL', 'NOT'):
            value = self.truth(self.pop(stack))
            expr = value if op == 'CAST_BOOL' else Expr(BOOL, 'not', None, (value,))
        elif op == 'CAST_INT':
            value = self.pop(stack)
            require(value.tag in (INT, BOOL), 'require exact int/bool cast operands')
            expr = value if value.tag == INT else Expr(INT, 'bool_int', None, (value,))
        elif op in ('AND', 'OR'):
            right, left = self.truth(self.pop(stack)), self.truth(self.pop(stack))
            expr = Expr(BOOL, 'binary', 'and' if op == 'AND' else 'or', (left, right))
        elif op in ('BOOL_AND', 'BOOL_OR'):
            right, left = self.pop(stack, BOOL), self.pop(stack, BOOL)
            expr = Expr(BOOL, 'binary', 'and' if op == 'BOOL_AND' else 'or', (left, right))
        elif op == 'BOOL_NOT':
            expr = Expr(BOOL, 'not', None, (self.pop(stack, BOOL),))
        elif op == 'CALL':
            require(not pure, 'require side-effect-free loop conditions without calls')
            require(0 <= arg < len(self.module['functions']), 'require an existing direct callee')
            callee = self.module['functions'][arg]
            args = [self.pop(stack, t) for t in reversed(callee['params'])]
            expr = Expr(callee['result'], 'call', arg, tuple(reversed(args)))
            self.calls.add(arg)
        else:
            raise Refusal('I require a simple scalar instruction here')
        if pure:
            stack.append(expr)
        else:
            temporary = Expr(expr.tag, 'temporary', ins['pc'])
            statements.append(('let', temporary, expr))
            stack.append(temporary)

    def region(self, start, end, initialized, depth=0):
        require(depth <= 32, 'limit structured region nesting to 32')
        statements, stack = [], []
        initialized = set(initialized)
        index = start
        while index < end:
            if index in self.backedges:
                require(not stack, 'require an empty loop-entry stack')
                back = self.backedges[index]
                require(index < back < end, 'require a loop wholly inside its enclosing region')
                branch = next((n for n in range(index, back) if self.code[n]['op'] in BRANCH), None)
                require(branch is not None, 'require a pretest condition')
                require(self.positions[self.code[branch]['target']] == back + 1,
                        'require the loop condition to exit immediately after its backedge')
                condition_stack = []
                for n in range(index, branch):
                    self.simple(n, condition_stack, initialized, [], pure=True)
                condition = self.pop(condition_stack, BOOL)
                require(not condition_stack, 'require one loop condition value')
                if self.code[branch]['op'] == 'JMP_TRUE':
                    condition = Expr(BOOL, 'not', None, (condition,))
                self.touch(branch)
                body, _, returns = self.region(branch + 1, back, initialized, depth + 1)
                require(not returns, 'do not reconstruct terminal loop bodies yet')
                self.touch(back)
                statements.append(('while', condition, body))
                index = back + 1
                continue
            ins = self.code[index]
            op = ins['op']
            if op in BRANCH:
                self.touch(index)
                condition = self.pop(stack, BOOL)
                require(not stack, 'require empty stacks at conditional region boundaries')
                target = self.positions[ins['target']]
                require(index < target <= end, 'require forward conditional regions')
                if op == 'JMP_TRUE':
                    condition = Expr(BOOL, 'not', None, (condition,))
                separator = target - 1
                has_else = separator > index and self.code[separator]['op'] == 'JMP'
                join = self.positions[self.code[separator]['target']] if has_else else target
                require(target <= join <= end, 'require a non-crossing forward diamond')
                left, left_init, left_ret = self.region(index + 1, separator if has_else else target,
                                                       initialized, depth + 1)
                if has_else:
                    self.touch(separator)
                    right, right_init, right_ret = self.region(target, join, initialized, depth + 1)
                else:
                    right, right_init, right_ret = [], initialized, False
                statements.append(('if', condition, left, right))
                if left_ret and right_ret:
                    require(join == end, 'do not discard code after a terminal region')
                    return statements, initialized, True
                initialized = right_init if left_ret else left_init if right_ret else left_init & right_init
                index = join
                continue
            if op == 'RET':
                self.touch(index)
                value = self.pop(stack, self.fn['result'])
                require(not stack and index + 1 == end, 'require explicit terminal scalar returns')
                statements.append(('return', value))
                return statements, initialized, True
            require(op != 'JMP', 'do not reconstruct an unstructured jump')
            self.simple(index, stack, initialized, statements)
            index += 1
        require(not stack, 'require empty stacks at region exits')
        return statements, initialized, False

    def run(self):
        body, _, returns = self.region(0, len(self.code), set(self.types))
        require(returns, 'require every function path to return explicitly')
        require(len(self.seen) == len(self.code), 'require complete instruction coverage')
        return Function(self.fn['name'], self.fn['params'], self.fn['result'],
                        self.types, self.fn['names'], body), self.calls


def analyze(module):
    require(0 < len(module['functions']) <= 32, 'limit reconstructed function count to 32')
    functions, calls = [], []
    for index in range(len(module['functions'])):
        function, callees = Analyze(module, index).run()
        functions.append(function)
        calls.append(callees)
    done, active = set(), set()

    def visit(index):
        require(index not in active, 'require an acyclic direct-call graph')
        if index in done:
            return
        active.add(index)
        for callee in calls[index]:
            visit(callee)
        active.remove(index)
        done.add(index)

    for index in range(len(functions)):
        visit(index)
    return functions


class Emit:
    def __init__(self, functions, language):
        self.functions, self.language = functions, language
        self.lines = []
        self.function = None

    def name(self, index):
        return f'nlr_f{index}_{self.functions[index].name or "unnamed"}'

    def local(self, slot):
        suffix = self.function.names[slot]
        return f'nlr_l{slot}' + ('_' + suffix if suffix else '')

    def type(self, tag):
        return ('int64_t' if tag == INT else 'bool') if self.language == 'c' else ('int' if tag == INT else 'bool')

    def expression(self, expr):
        if expr.kind == 'constant':
            if expr.tag == BOOL:
                return 'true' if expr.value else 'false'
            if self.language == 'c':
                return 'INT64_MIN' if expr.value == -(1 << 63) else f'INT64_C({expr.value})'
            return '(- -9223372036854775807 1)' if expr.value == -(1 << 63) else str(expr.value)
        if expr.kind == 'local':
            return self.local(expr.value)
        if expr.kind == 'temporary':
            return f'nlr_t{expr.value}'
        args = [self.expression(a) for a in expr.args]
        if expr.kind in ('arithmetic', 'unsigned_compare'):
            name = 'nlr_i64_' + expr.value
            return name + '(' + ', '.join(args) + ')' if self.language == 'c' else '(' + ' '.join([name] + args) + ')'
        if expr.kind == 'call':
            name = self.name(expr.value)
            return name + '(' + ', '.join(args) + ')' if self.language == 'c' else '(' + ' '.join([name] + args) + ')'
        if expr.kind == 'bool_int':
            return '((int64_t)' + args[0] + ')' if self.language == 'c' else '(nlr_bool_int ' + args[0] + ')'
        if expr.kind == 'not':
            return '(!' + args[0] + ')' if self.language == 'c' else '(not ' + args[0] + ')'
        op = expr.value
        if self.language == 'c':
            op = {'and': '&&', 'or': '||'}.get(op, op)
            return '(' + args[0] + ' ' + op + ' ' + args[1] + ')'
        return '(' + ' '.join([op] + args) + ')'

    def line(self, value, indent=0):
        self.lines.append('    ' * indent + value)

    def statements(self, statements, indent=1):
        c = self.language == 'c'
        for stmt in statements:
            kind = stmt[0]
            if kind == 'let':
                _, target, value = stmt
                name, expression = self.expression(target), self.expression(value)
                self.line(f'{self.type(target.tag)} {name} = {expression};' if c else
                          f'let {name}: {self.type(target.tag)} = {expression}', indent)
                if c:
                    self.line(f'(void){name};', indent)
            elif kind == 'store':
                self.line(f'{self.local(stmt[1])} = {self.expression(stmt[2])};' if c else
                          f'set {self.local(stmt[1])} {self.expression(stmt[2])}', indent)
            elif kind == 'return':
                self.line('return ' + self.expression(stmt[1]) + (';' if c else ''), indent)
            elif kind in ('if', 'while'):
                expression = self.expression(stmt[1])
                self.line(f'{kind} ({expression}) {{' if c else f'{kind} {expression} {{', indent)
                self.statements(stmt[2], indent + 1)
                if kind == 'if' and stmt[3]:
                    self.line('} else {', indent)
                    self.statements(stmt[3], indent + 1)
                self.line('}', indent)

    def signature(self, index):
        function = self.functions[index]
        if self.language == 'c':
            args = ', '.join(f'{self.type(t)} nlr_a{p}' for p, t in enumerate(function.params)) or 'void'
            return f'{self.type(function.result)} {self.name(index)}({args})'
        args = ', '.join(f'nlr_a{p}: {self.type(t)}' for p, t in enumerate(function.params))
        return f'fn {self.name(index)}({args}) -> {self.type(function.result)}'

    def run(self, entry):
        c = self.language == 'c'
        if c:
            self.line('#include <stdint.h>\n#include <stdbool.h>\n#include <inttypes.h>')
            for index in range(len(self.functions)):
                self.line(self.signature(index) + ';')
        else:
            self.line('# I reconstruct executable scalar regions; original shadows are not retained.')
        self.arithmetic_helpers()
        for index, function in enumerate(self.functions):
            self.function = function
            self.line(self.signature(index) + ' {')
            for slot, tag in sorted(function.locals.items()):
                value = f'nlr_a{slot}' if slot < len(function.params) else '0' if tag == INT else 'false'
                name = self.local(slot)
                self.line(f'{self.type(tag)} {name} = {value};' if c else f'let mut {name}: {self.type(tag)} = {value}', 1)
                if c:
                    self.line(f'(void){name};', 1)
            self.statements(function.body)
            self.line('}')
        self.line('int main(void) {' if c else 'fn main() -> int {')
        self.line(f'return (int){self.name(entry)}();' if c else f'return ({self.name(entry)})', 1)
        self.line('}')
        return '\n'.join(self.lines) + '\n'

    def arithmetic_helpers(self):
        needed = set()

        def collect(node):
            if isinstance(node, Expr):
                if node.kind in ('arithmetic', 'unsigned_compare'):
                    needed.add(node.value)
                if node.kind == 'bool_int':
                    needed.add('bool_int')
                collect(node.args)
            elif isinstance(node, (tuple, list)):
                for child in node:
                    collect(child)

        for function in self.functions:
            collect(function.body)
        if 'bool_int' in needed:
            needed.remove('bool_int')
            if self.language != 'c':
                self.line('''fn nlr_bool_int(value: bool) -> int {
    if value { return 1 }
    return 0
}
shadow nlr_bool_int {
    assert (== (nlr_bool_int false) 0)
    assert (== (nlr_bool_int true) 1)
}''')
        if not needed:
            return
        if 'mul_high_s' in needed:
            needed.add('mul_high_u')
        if self.language == 'c':
            if needed & {'add', 'sub', 'mul', 'neg', 'shl', 'shr_s', 'shr_u', 'band', 'bor', 'bxor', 'invert', 'div_u', 'rem_u', 'mul_high_s', 'mul_high_u'}:
                self.line('''static int64_t nlr_i64_bits(uint64_t bits) {
    if (bits <= (uint64_t)INT64_MAX) return (int64_t)bits;
    return -INT64_C(1) - (int64_t)(UINT64_MAX - bits);
}''')
            for op in sorted(needed, key=lambda op: (op == 'mul_high_s', op)):
                args = 'int64_t a' if op in ('neg', 'invert') else 'int64_t a, int64_t b'
                if op == 'mul_high_u':
                    self.line('static int64_t nlr_i64_mul_high_u(int64_t a, int64_t b) {')
                    for operand in ('a', 'b'):
                        for limb in range(4):
                            self.line(f'    uint64_t {operand}{limb} = ((uint64_t){operand} >> {16 * limb}) & UINT64_C(65535);')
                    for digit in range(7):
                        terms = [f'a{i} * b{digit-i}' for i in range(4) if 0 <= digit-i < 4]
                        if digit:
                            terms.insert(0, f'(c{digit-1} >> 16)')
                        self.line(f'    uint64_t c{digit} = ' + ' + '.join(terms) + ';')
                    self.line('    uint64_t high = (c4 & UINT64_C(65535)) | ((c5 & UINT64_C(65535)) << 16) | ((c6 & UINT64_C(65535)) << 32) | ((c6 >> 16) << 48);')
                    self.line('    return nlr_i64_bits(high);\n}')
                    continue
                if op == 'mul_high_s':
                    self.line('''static int64_t nlr_i64_mul_high_s(int64_t a, int64_t b) {
    uint64_t high = (uint64_t)nlr_i64_mul_high_u(a, b);
    if (a < 0) high -= (uint64_t)b;
    if (b < 0) high -= (uint64_t)a;
    return nlr_i64_bits(high);
}''')
                    continue
                if op in ('lt_u', 'le_u', 'gt_u', 'ge_u'):
                    symbol = {'lt_u': '<', 'le_u': '<=', 'gt_u': '>', 'ge_u': '>='}[op]
                    self.line(f'static bool nlr_i64_{op}({args}) {{ return (uint64_t)a {symbol} (uint64_t)b; }}')
                    continue
                if op in ('shl', 'shr_s', 'shr_u'):
                    operator = '<<' if op == 'shl' else '>>'
                    self.line(f'''static int64_t nlr_i64_{op}({args}) {{
    unsigned count = (unsigned)((uint64_t)b & UINT64_C(63));
    uint64_t bits = (uint64_t)a {operator} count;''')
                    if op == 'shr_s':
                        self.line('    if (a < 0 && count != 0) bits |= UINT64_MAX << (64U - count);')
                    self.line('    return nlr_i64_bits(bits);\n}')
                    continue
                if op in ('div_u', 'rem_u'):
                    symbol = '/' if op == 'div_u' else '%'
                    self.line(f'''static int64_t nlr_i64_{op}({args}) {{
    if (b == 0) return INT64_C(0);
    return nlr_i64_bits((uint64_t)a {symbol} (uint64_t)b);
}}''')
                    continue
                if op in ('div', 'rem'):
                    symbol = '/' if op == 'div' else '%'
                    overflow = 'INT64_MIN' if op == 'div' else 'INT64_C(0)'
                    self.line(f'''static int64_t nlr_i64_{op}({args}) {{
    if (b == 0) return INT64_C(0);
    if (a == INT64_MIN && b == -1) return {overflow};
    return a {symbol} b;
}}''')
                    continue
                expression = {'add': '(uint64_t)a + (uint64_t)b',
                              'sub': '(uint64_t)a - (uint64_t)b',
                              'mul': '(uint64_t)a * (uint64_t)b',
                              'neg': 'UINT64_C(0) - (uint64_t)a',
                              'band': '(uint64_t)a & (uint64_t)b',
                              'bor': '(uint64_t)a | (uint64_t)b',
                              'bxor': '(uint64_t)a ^ (uint64_t)b',
                              'invert': '~(uint64_t)a'}[op]
                self.line(f'static int64_t nlr_i64_{op}({args}) {{ return nlr_i64_bits({expression}); }}')
            return
        if needed & {'div_u', 'rem_u'}:
            needed.update(('add', 'sub', 'ge_u'))
        if needed & {'band', 'bor', 'bxor'}:
            needed.update(('add', 'shr_u'))
        if 'mul_high_u' in needed:
            needed.update(('add', 'shl', 'shr_u'))
        if 'mul_high_s' in needed:
            needed.add('sub')
        if 'shl' in needed:
            needed.add('add')
        if 'mul' in needed:
            needed.update(('add', 'sub'))
        for op in sorted(needed, key=lambda op: (op == 'mul_high_s', op)):
            self.line(NANO_INTEGER_HELPERS[op])


# I branch before signed arithmetic so all helper intermediates are representable.
# These shadows test my helper implementation, not an original source harness.
NANO_INTEGER_HELPERS = {
    'ge_u': '''fn nlr_i64_ge_u(a: int, b: int) -> bool {
    let a_negative: bool = (< a 0)
    let b_negative: bool = (< b 0)
    if (!= a_negative b_negative) { return a_negative }
    return (>= a b)
}
shadow nlr_i64_ge_u {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_ge_u 0 -1) false)
    assert (== (nlr_i64_ge_u low -1) false)
    assert (== (nlr_i64_ge_u -1 low) true)
    assert (== (nlr_i64_ge_u 7 7) true)
}''',
    'gt_u': '''fn nlr_i64_gt_u(a: int, b: int) -> bool {
    let a_negative: bool = (< a 0)
    let b_negative: bool = (< b 0)
    if (!= a_negative b_negative) { return a_negative }
    return (> a b)
}
shadow nlr_i64_gt_u {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_gt_u 0 -1) false)
    assert (== (nlr_i64_gt_u low -1) false)
    assert (== (nlr_i64_gt_u -1 low) true)
    assert (== (nlr_i64_gt_u 7 7) false)
}''',
    'le_u': '''fn nlr_i64_le_u(a: int, b: int) -> bool {
    let a_negative: bool = (< a 0)
    let b_negative: bool = (< b 0)
    if (!= a_negative b_negative) { return b_negative }
    return (<= a b)
}
shadow nlr_i64_le_u {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_le_u 0 -1) true)
    assert (== (nlr_i64_le_u low -1) true)
    assert (== (nlr_i64_le_u -1 low) false)
    assert (== (nlr_i64_le_u 7 7) true)
}''',
    'lt_u': '''fn nlr_i64_lt_u(a: int, b: int) -> bool {
    let a_negative: bool = (< a 0)
    let b_negative: bool = (< b 0)
    if (!= a_negative b_negative) { return b_negative }
    return (< a b)
}
shadow nlr_i64_lt_u {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_lt_u 0 -1) true)
    assert (== (nlr_i64_lt_u low -1) true)
    assert (== (nlr_i64_lt_u -1 low) false)
    assert (== (nlr_i64_lt_u 7 7) false)
}''',
    'invert': '''fn nlr_i64_invert(a: int) -> int {
    return (- -1 a)
}
shadow nlr_i64_invert {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_invert low) 9223372036854775807)
    assert (== (nlr_i64_invert 9223372036854775807) low)
    assert (== (nlr_i64_invert 0) -1)
    assert (== (nlr_i64_invert -1) 0)
}''',
    'bxor': '''fn nlr_i64_bxor(a: int, b: int) -> int {
    let mut left: int = a
    let mut right: int = b
    let mut weight: int = 1
    let mut result: int = 0
    let mut step: int = 0
    while (< step 64) {
        let left_bit: bool = (!= (% left 2) 0)
        let right_bit: bool = (!= (% right 2) 0)
        if (!= left_bit right_bit) { set result (nlr_i64_add result weight) }
        set left (nlr_i64_shr_u left 1)
        set right (nlr_i64_shr_u right 1)
        set weight (nlr_i64_add weight weight)
        set step (+ step 1)
    }
    return result
}
shadow nlr_i64_bxor {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_bxor 6 3) 5)
    assert (== (nlr_i64_bxor low -1) 9223372036854775807)
}''',
    'bor': '''fn nlr_i64_bor(a: int, b: int) -> int {
    let mut left: int = a
    let mut right: int = b
    let mut weight: int = 1
    let mut result: int = 0
    let mut step: int = 0
    while (< step 64) {
        let left_bit: bool = (!= (% left 2) 0)
        let right_bit: bool = (!= (% right 2) 0)
        if (or left_bit right_bit) { set result (nlr_i64_add result weight) }
        set left (nlr_i64_shr_u left 1)
        set right (nlr_i64_shr_u right 1)
        set weight (nlr_i64_add weight weight)
        set step (+ step 1)
    }
    return result
}
shadow nlr_i64_bor {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_bor 6 3) 7)
    assert (== (nlr_i64_bor low -1) -1)
}''',
    'band': '''fn nlr_i64_band(a: int, b: int) -> int {
    let mut left: int = a
    let mut right: int = b
    let mut weight: int = 1
    let mut result: int = 0
    let mut step: int = 0
    while (< step 64) {
        let left_bit: bool = (!= (% left 2) 0)
        let right_bit: bool = (!= (% right 2) 0)
        if (and left_bit right_bit) { set result (nlr_i64_add result weight) }
        set left (nlr_i64_shr_u left 1)
        set right (nlr_i64_shr_u right 1)
        set weight (nlr_i64_add weight weight)
        set step (+ step 1)
    }
    return result
}
shadow nlr_i64_band {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_band 6 3) 2)
    assert (== (nlr_i64_band low -1) low)
}''',
    'shr_u': '''fn nlr_i64_shr_u(a: int, b: int) -> int {
    let mut count: int = (% b 64)
    if (< count 0) { set count (+ count 64) }
    let mut result: int = a
    let mut step: int = 0
    while (< step count) {
        let mut half: int = (/ result 2)
        if (and (< result 0) (!= (% result 2) 0)) { set half (- half 1) }
        if (< result 0) { set half (+ (+ half 9223372036854775807) 1) }
        set result half
        set step (+ step 1)
    }
    return result
}
shadow nlr_i64_shr_u {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_shr_u -1 1) 9223372036854775807)
    assert (== (nlr_i64_shr_u low -1) 1)
    assert (== (nlr_i64_shr_u low 64) low)
    assert (== (nlr_i64_shr_u 8 65) 4)
}''',
    'shr_s': '''fn nlr_i64_shr_s(a: int, b: int) -> int {
    let mut count: int = (% b 64)
    if (< count 0) { set count (+ count 64) }
    let mut result: int = a
    let mut step: int = 0
    while (< step count) {
        let mut half: int = (/ result 2)
        if (and (< result 0) (!= (% result 2) 0)) { set half (- half 1) }
        set result half
        set step (+ step 1)
    }
    return result
}
shadow nlr_i64_shr_s {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_shr_s -7 1) -4)
    assert (== (nlr_i64_shr_s low -1) -1)
    assert (== (nlr_i64_shr_s low 64) low)
    assert (== (nlr_i64_shr_s 8 65) 4)
}''',
    'shl': '''fn nlr_i64_shl(a: int, b: int) -> int {
    let mut count: int = (% b 64)
    if (< count 0) { set count (+ count 64) }
    let mut result: int = a
    let mut step: int = 0
    while (< step count) {
        set result (nlr_i64_add result result)
        set step (+ step 1)
    }
    return result
}
shadow nlr_i64_shl {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_shl 1 -1) low)
    assert (== (nlr_i64_shl low 1) 0)
    assert (== (nlr_i64_shl low 64) low)
    assert (== (nlr_i64_shl 8 65) 16)
}''',
    'rem': '''fn nlr_i64_rem(a: int, b: int) -> int {
    if (== b 0) { return 0 }
    let low: int = (- -9223372036854775807 1)
    if (and (== a low) (== b -1)) { return 0 }
    return (% a b)
}
shadow nlr_i64_rem {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_rem low -1) 0)
    assert (== (nlr_i64_rem low 0) 0)
    assert (== (nlr_i64_rem -7 3) -1)
    assert (== (nlr_i64_rem 7 -3) 1)
}''',
    'div': '''fn nlr_i64_div(a: int, b: int) -> int {
    if (== b 0) { return 0 }
    let low: int = (- -9223372036854775807 1)
    if (and (== a low) (== b -1)) { return low }
    return (/ a b)
}
shadow nlr_i64_div {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_div low -1) low)
    assert (== (nlr_i64_div low 0) 0)
    assert (== (nlr_i64_div -7 3) -2)
    assert (== (nlr_i64_div 7 -3) -2)
}''',
    'mul': '''fn nlr_i64_mul(a: int, b: int) -> int {
    let mut factor: int = a
    let mut remaining: int = b
    let mut result: int = 0
    while (!= remaining 0) {
        let digit: int = (% remaining 2)
        if (> digit 0) { set result (nlr_i64_add result factor) }
        if (< digit 0) { set result (nlr_i64_sub result factor) }
        set remaining (/ remaining 2)
        set factor (nlr_i64_add factor factor)
    }
    return result
}
shadow nlr_i64_mul {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_mul low -1) low)
    assert (== (nlr_i64_mul low 2) 0)
    assert (== (nlr_i64_mul 9223372036854775807 9223372036854775807) 1)
    assert (== (nlr_i64_mul -3 -7) 21)
    assert (== (nlr_i64_mul -3 7) -21)
    assert (== (nlr_i64_mul 4294967296 4294967296) 0)
}''',
    'add': '''fn nlr_i64_add(a: int, b: int) -> int {
    let low: int = (- -9223372036854775807 1)
    let high: int = 9223372036854775807
    if (> b 0) {
        let boundary: int = (- high b)
        if (> a boundary) { return (+ low (- (- a boundary) 1)) }
    }
    if (< b 0) {
        let boundary: int = (- low b)
        if (< a boundary) { return (+ high (+ (- a boundary) 1)) }
    }
    return (+ a b)
}
shadow nlr_i64_add {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_add 9223372036854775807 1) low)
    assert (== (nlr_i64_add low -1) 9223372036854775807)
    assert (== (nlr_i64_add low low) 0)
    assert (== (nlr_i64_add 7 -3) 4)
}''',
    'sub': '''fn nlr_i64_sub(a: int, b: int) -> int {
    let low: int = (- -9223372036854775807 1)
    let high: int = 9223372036854775807
    if (> b 0) {
        let boundary: int = (+ low b)
        if (< a boundary) { return (+ high (+ (- a boundary) 1)) }
    }
    if (< b 0) {
        let boundary: int = (+ high b)
        if (> a boundary) { return (+ low (- a (+ boundary 1))) }
    }
    return (- a b)
}
shadow nlr_i64_sub {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_sub low 1) 9223372036854775807)
    assert (== (nlr_i64_sub 9223372036854775807 -1) low)
    assert (== (nlr_i64_sub 9223372036854775807 low) -1)
    assert (== (nlr_i64_sub low low) 0)
}''',
    'neg': '''fn nlr_i64_neg(a: int) -> int {
    let low: int = (- -9223372036854775807 1)
    if (== a low) { return low }
    return (- 0 a)
}
shadow nlr_i64_neg {
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_neg low) low)
    assert (== (nlr_i64_neg 9223372036854775807) -9223372036854775807)
    assert (== (nlr_i64_neg -1) 1)
    assert (== (nlr_i64_neg 0) 0)
}''',
}


# I consume the dividend from its high bit. A saved remainder carry keeps
# unsigned subtraction correct even when doubled remainder wraps its carrier.
for _operation, _result in (('div_u', 'quotient'), ('rem_u', 'remainder')):
    _quotient_init = '    let mut quotient: int = 0\n' if _result == 'quotient' else ''
    _quotient_shift = '        set quotient (nlr_i64_add quotient quotient)\n' if _result == 'quotient' else ''
    _quotient_bit = '            set quotient (nlr_i64_add quotient 1)\n' if _result == 'quotient' else ''
    NANO_INTEGER_HELPERS[_operation] = f'''fn nlr_i64_{_operation}(a: int, b: int) -> int {{
    if (== b 0) {{ return 0 }}
    let mut dividend: int = a
    let mut remainder: int = 0
{_quotient_init}    let mut step: int = 0
    while (< step 64) {{
        let carry: bool = (< remainder 0)
        set remainder (nlr_i64_add remainder remainder)
        if (< dividend 0) {{ set remainder (nlr_i64_add remainder 1) }}
        set dividend (nlr_i64_add dividend dividend)
{_quotient_shift}        let at_least: bool = (nlr_i64_ge_u remainder b)
        if (or carry at_least) {{
            set remainder (nlr_i64_sub remainder b)
{_quotient_bit}        }}
        set step (+ step 1)
    }}
    return {_result}
}}
shadow nlr_i64_{_operation} {{
    let low: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_{_operation} -1 low) {1 if _result == 'quotient' else 9223372036854775807})
    assert (== (nlr_i64_{_operation} low 0) 0)
    assert (== (nlr_i64_{_operation} -1 1) {-1 if _result == 'quotient' else 0})
    assert (== (nlr_i64_{_operation} 7 3) {2 if _result == 'quotient' else 1})
}}'''


# I use base-65536 convolution: products <2^32, coefficient+carry <2^34.
# Normalizing a signed remainder yields the low unsigned limb because 2^64
# is divisible by 65536. Higher limbs come from total logical shifts.
_wide_unsigned = ['fn nlr_i64_mul_high_u(a: int, b: int) -> int {']
for _operand in ('a', 'b'):
    _wide_unsigned.append(f'    let mut {_operand}0: int = (% {_operand} 65536)')
    _wide_unsigned.append(f'    if (< {_operand}0 0) {{ set {_operand}0 (+ {_operand}0 65536) }}')
    for _limb in range(1, 4):
        _wide_unsigned.append(f'    let {_operand}{_limb}: int = (% (nlr_i64_shr_u {_operand} {16 * _limb}) 65536)')
for _digit in range(7):
    _terms = [f'(* a{i} b{_digit-i})' for i in range(4) if 0 <= _digit-i < 4]
    if _digit:
        _terms.insert(0, f'(/ c{_digit-1} 65536)')
    _sum = _terms[0]
    for _term in _terms[1:]:
        _sum = f'(+ {_sum} {_term})'
    _wide_unsigned.append(f'    let c{_digit}: int = {_sum}')
_wide_unsigned.extend([
    '    let low: int = (nlr_i64_add (% c4 65536) (nlr_i64_shl (% c5 65536) 16))',
    '    let high: int = (nlr_i64_add (nlr_i64_shl (% c6 65536) 32) (nlr_i64_shl (/ c6 65536) 48))',
    '    return (nlr_i64_add low high)',
    '}',
    'shadow nlr_i64_mul_high_u {',
    '    assert (== (nlr_i64_mul_high_u -1 -1) -2)',
    '    assert (== (nlr_i64_mul_high_u 4294967296 4294967296) 1)',
    '    assert (== (nlr_i64_mul_high_u 0 -1) 0)',
    '}',
])
NANO_INTEGER_HELPERS['mul_high_u'] = '\n'.join(_wide_unsigned)
NANO_INTEGER_HELPERS['mul_high_s'] = '''fn nlr_i64_mul_high_s(a: int, b: int) -> int {
    let mut high: int = (nlr_i64_mul_high_u a b)
    if (< a 0) { set high (nlr_i64_sub high b) }
    if (< b 0) { set high (nlr_i64_sub high a) }
    return high
}
shadow nlr_i64_mul_high_s {
    let minimum: int = (- -9223372036854775807 1)
    assert (== (nlr_i64_mul_high_s -1 -1) 0)
    assert (== (nlr_i64_mul_high_s minimum 2) -1)
    assert (== (nlr_i64_mul_high_s 4294967296 4294967296) 1)
}'''
