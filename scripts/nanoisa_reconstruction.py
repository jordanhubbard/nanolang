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
BRANCH = {'JMP_TRUE', 'JMP_FALSE'}
ARITHMETIC = {'I64_ADD': 'add', 'I64_SUB': 'sub', 'I64_NEG': 'neg', 'I64_MUL': 'mul'}
SIMPLE = {'NOP', 'PUSH_I64', 'PUSH_BOOL', 'LOAD_LOCAL', 'STORE_LOCAL',
          'DUP', 'POP', 'SWAP', 'BOOL_AND', 'BOOL_OR', 'BOOL_NOT', 'CALL'} | set(COMPARE) | set(ARITHMETIC)


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
        elif op == 'SWAP':
            right, left = self.pop(stack), self.pop(stack)
            stack.extend((right, left))
            return
        elif op in ARITHMETIC:
            right = self.pop(stack, INT)
            args = (right,) if op == 'I64_NEG' else (self.pop(stack, INT), right)
            expr = Expr(INT, 'arithmetic', ARITHMETIC[op], args)
        elif op in COMPARE:
            right, left = self.pop(stack, INT), self.pop(stack, INT)
            expr = Expr(BOOL, 'binary', COMPARE[op], (left, right))
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
        if expr.kind == 'arithmetic':
            name = 'nlr_i64_' + expr.value
            return name + '(' + ', '.join(args) + ')' if self.language == 'c' else '(' + ' '.join([name] + args) + ')'
        if expr.kind == 'call':
            name = self.name(expr.value)
            return name + '(' + ', '.join(args) + ')' if self.language == 'c' else '(' + ' '.join([name] + args) + ')'
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
                if node.kind == 'arithmetic':
                    needed.add(node.value)
                collect(node.args)
            elif isinstance(node, (tuple, list)):
                for child in node:
                    collect(child)

        for function in self.functions:
            collect(function.body)
        if not needed:
            return
        if self.language == 'c':
            self.line('''static int64_t nlr_i64_bits(uint64_t bits) {
    if (bits <= (uint64_t)INT64_MAX) return (int64_t)bits;
    return -INT64_C(1) - (int64_t)(UINT64_MAX - bits);
}''')
            for op in sorted(needed):
                args = 'int64_t a' if op == 'neg' else 'int64_t a, int64_t b'
                expression = {'add': '(uint64_t)a + (uint64_t)b',
                              'sub': '(uint64_t)a - (uint64_t)b',
                              'mul': '(uint64_t)a * (uint64_t)b',
                              'neg': 'UINT64_C(0) - (uint64_t)a'}[op]
                self.line(f'static int64_t nlr_i64_{op}({args}) {{ return nlr_i64_bits({expression}); }}')
            return
        if 'mul' in needed:
            needed.update(('add', 'sub'))
        for op in sorted(needed):
            self.line(NANO_INTEGER_HELPERS[op])


# I branch before signed arithmetic so all helper intermediates are representable.
# These shadows test my helper implementation, not an original source harness.
NANO_INTEGER_HELPERS = {
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
