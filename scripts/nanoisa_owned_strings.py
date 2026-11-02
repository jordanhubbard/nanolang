"""I retain bounded executable literal bytes, never advisory string contents."""
from nanoisa_reconstruction import Expr, require

STRING = 5
NO_INDEX = 4294967295
LITERAL_BYTES = 1048576


def literal_facts(module):
    functions = module['functions']
    referenced = set()
    uses = any(pair[0] == STRING for fn in functions for pair in fn['params'])
    for fn in functions:
        for ins in fn['code']:
            if ins['op'] == 'PUSH_STR':
                args = ins.get('args')
                require(isinstance(args, list) and len(args) == 1 and
                        type(args[0]) is int and type(ins['arg']) is int and args[0] == ins['arg'],
                        'require one exact executable string index')
                referenced.add(args[0]); uses = True
            elif ins['op'] in ('PRINT', 'PRINTLN'):
                uses = True
    if not uses:
        require('string_literals' not in module and 'string_constant_count' not in module,
                'refuse unreferenced executable string facts')
        return {}
    require(not any(any(fn.get('param_modes', [])) for fn in functions),
            'refuse STRING in borrowed reconstruction profiles')
    count, rows = module.get('string_constant_count'), module.get('string_literals')
    require(type(count) is int and 0 <= count <= NO_INDEX and
            isinstance(rows, list) and len(rows) <= min(4096, count) and
            len(rows) == len(referenced), 'require bounded exact executable string facts')
    result, total = {}, 0
    for row in rows:
        require(isinstance(row, dict), 'require exact executable string rows')
        index, length, hex_bytes = row.get('index'), row.get('length'), row.get('hex')
        require(type(index) is int and index in referenced and 0 <= index < count and
                index not in result and type(length) is int and 0 <= length <= LITERAL_BYTES - total and
                isinstance(hex_bytes, str) and len(hex_bytes) == 2 * length and
                all(c in '0123456789abcdef' for c in hex_bytes),
                'require exact bounded string indexes lengths and hexadecimal bytes')
        total += length
        value = bytes.fromhex(hex_bytes)
        require(b'\0' not in value, 'refuse embedded NUL executable strings')
        result[index] = value
    require(set(result) == referenced, 'require every executable string literal')
    return result


def nano_literal(value):
    try:
        text = value.decode('utf-8', 'strict')
    except UnicodeDecodeError:
        require(False, 'require UTF-8 literal bytes for Nano source output')
    escaped = {'\\': '\\\\', '"': '\\"', '\n': '\\n', '\r': '\\r', '\t': '\\t'}
    return '"' + ''.join(escaped.get(c, c) for c in text) + '"'


def literals_in(functions):
    found = {}
    def visit(node):
        if isinstance(node, Expr):
            if node.kind == 'string_literal':
                index, data = node.value
                require(index not in found or found[index] == data, 'require immutable indexed literal bytes')
                found[index] = data
            for child in node.args: visit(child)
        elif isinstance(node, (tuple, list)):
            for child in node: visit(child)
    for fn in functions: visit(fn.body)
    return found
