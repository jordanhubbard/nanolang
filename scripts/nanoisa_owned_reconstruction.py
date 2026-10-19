"""I retain ordered local resource transfers in one bounded owned region tree."""
from nanoisa_reconstruction import Analyze, Emit, Expr, Function, INT, BOOL, SIMPLE, require


from nanoisa_owned_strings import STRING, literal_facts, nano_literal, literals_in

VOID = 0

# I inline only nontrapping scalar pretests without duplicated/discarded operands.
OWNED_PRETEST = {
    "NOP", "PUSH_I64", "PUSH_BOOL", "LOAD_LOCAL",
    "EQ", "NE", "LT", "LE", "GT", "GE",
    "I64_EQ", "I64_NE", "I64_LT_S", "I64_LE_S", "I64_GT_S", "I64_GE_S",
    "I64_LT_U", "I64_LE_U", "I64_GT_U", "I64_GE_U",
    "BOOL_AND", "BOOL_OR", "BOOL_NOT", "AND", "OR", "NOT", "CAST_BOOL", "CAST_INT",
}


# I materialize only the source-qualified exact-INT wrapping operations.
OWNED_WRAPPING_PRETEST = {"ADD", "SUB", "MUL"}


def owned_type(layout):
    return ('owner', layout)


class OwnedAnalyze(Analyze):
    def __init__(self, module, index=0):
        self.module = module
        require(module.get('profile') == 'owned-local-v1' and module['entry'] == 0 and
                0 < len(module['functions']) <= 8, 'require bounded verified resource functions')
        self.string_literals = literal_facts(module)
        self.borrowed = any(any(f.get('param_modes', [])) for f in module['functions'])
        self.fn = module['functions'][index]
        self.index = index
        self.layouts = module['layouts']
        require(0 < len(self.layouts) <= 16, 'require bounded resource layouts')
        self.fields = []
        self.depths = []
        for i, layout in enumerate(self.layouts):
            require(layout['identity'] == i and layout['resource'] is True and
                    0 < len(layout['fields']) <= 16 and
                    all(t in (INT, BOOL, 8) for t in layout['fields']),
                    'require exact resource identity and bounded fields')
            fields = layout.get('field_types', [[t, 4294967295] for t in layout['fields']])
            require(isinstance(fields, list) and
                    all(isinstance(pair, list) and len(pair) == 2 and
                        all(type(value) is int for value in pair) for pair in fields) and
                    len(fields) == len(layout['fields']) and
                    all(pair[0] == tag for pair, tag in zip(fields, layout['fields'])),
                    'require complete exact resource field descriptors')
            depth = 1
            for tag, child in fields:
                if tag == 8:
                    require(0 <= child < i, 'require exact prior resource children')
                    depth = max(depth, self.depths[child] + 1)
                    require(depth <= 3, 'limit consuming resource layouts to depth three')
                else:
                    require(tag in (INT, BOOL) and child == 4294967295,
                            'require exact scalar fields without child layout identities')
            self.depths.append(depth)
            self.fields.append([owned_type(child) if tag == 8 else tag for tag, child in fields])
        self.declarations = self.fn['local_types']
        require(len(self.declarations) == self.fn['locals'] <= 128,
                'require complete local declarations')
        self.types = {i: t for i, (t, _) in enumerate(self.declarations) if t in (INT, BOOL)}

        def declared(pair, parameter=False):
            tag, layout = pair
            require((tag in (INT, BOOL) or parameter and not self.borrowed and tag == STRING) and layout == 4294967295 or
                    tag == 8 and 0 <= layout < len(self.layouts),
                    'require exact scalar or nominal resource declarations')
            return owned_type(layout) if tag == 8 else tag
        self.params = [declared(t, True) for t in self.fn['params']]
        self.string_params = {i for i, t in enumerate(self.params) if t == STRING}
        require(all(pair[0] != STRING or i in self.string_params and pair == [STRING, 4294967295]
                    for i, pair in enumerate(self.declarations)), 'require STRING only in exact parameter slots')
        self.modes = self.fn.get('param_modes', [0] * len(self.params))
        require(len(self.modes) == len(self.params) and all(m in (0, 1, 2) for m in self.modes),
                'require exact value or borrowed parameter modes')
        self.borrowed = any(any(f.get('param_modes', [])) for f in module['functions'])
        version = module.get('ownership_version', 1)
        require(version in (1, 2), 'require retained ownership version one or two')
        self.paths = version == 2
        if self.paths:
            require(all('field_types' in layout for layout in self.layouts),
                    'require complete exact path layout descriptors')
            require(self.borrowed and len(module['functions']) == 2 and
                    module['functions'][1].get('param_modes') in ([2, 2, 1], [1, 1], [1], [2]),
                    'require the exact nested-path helper profile')
        if self.borrowed:
            require(max(self.depths) <= (3 if self.paths else 2), 'retain nested borrowed modules at depth three' if self.paths else 'retain borrowed modules at depth two')
            require(len(module['functions']) == 2 and
                    module['functions'][0].get('param_modes', []) == [] and
                    module['functions'][1].get('param_modes') in (([2, 2, 1], [1, 1], [1], [2]) if self.paths else ([1], [2], [2, 1, 2])),
                    'require the separate entry-to-borrowed-helper profile')
        self.shared_paths = self.paths and module['functions'][1].get('param_modes') == [1, 1]
        self.single_shared_path = self.paths and module['functions'][1].get('param_modes') == [1]
        self.single_exclusive_path = self.paths and module['functions'][1].get('param_modes') == [2]
        self.exclusive_control = self.borrowed and module['functions'][1].get('param_modes') == [2] and (
            self.paths or any(ins['op'] in ('JMP', 'JMP_TRUE', 'JMP_FALSE')
                              for ins in module['functions'][1]['code']))
        from nanoisa_owned_exits import needs_loop_exits
        self.loop_exits = needs_loop_exits(module)
        if self.loop_exits:
            require(not self.paths and self.borrowed and len(self.layouts) == 1 and self.fields[0] == [INT, BOOL] and
                    module['functions'][1].get('param_modes') == [2],
                    'require the exact v1 single-exclusive scalar-leaf loop-exit profile')
        self.shared_root = None
        self.multi = self.borrowed and len(module['functions'][1].get('param_modes', [])) > 1
        self.exclusive = self.borrowed and 2 in module['functions'][1].get('param_modes', [])
        self.helper = self.borrowed and index == 1
        if self.helper:
            for slot, param in enumerate(self.params):
                require(isinstance(param, tuple) and all(t in (INT, BOOL) for t in self.fields[param[1]]),
                        'require exact scalar-leaf borrowed helper parameters')
                self.params[slot] = ('borrow', param[1], self.modes[slot])
        self.result = VOID if self.fn['result_type'] == [VOID, 4294967295] else declared(self.fn['result_type'])
        if self.loop_exits:
            require(self.result == INT, 'retain exact INT results in the loop-exit profile')
        require(self.fn.get('result_count') == (0 if self.result == VOID else 1),
                'require the exact owned helper result count')
        require(self.fn['result'] == self.fn['result_type'][0], 'require matching result declarations')
        require(len(self.params) <= 8 and len(self.params) <= len(self.declarations) and
                all(declared(self.declarations[i], True) == (owned_type(t[1]) if self.modes[i] else t) for i, t in enumerate(self.params)),
                'require exact initial parameter slots')
        require(index != 0 or not self.params and self.result == INT,
                'require a scalar entry without parameters')
        self.live = {i: t[1] for i, t in enumerate(self.params) if isinstance(t, tuple) and t[0] == 'owner'}
        self.used = set(self.live)
        self.seen = set()
        self.calls = set()
        self.code = self.fn['code']
        require(not any(ins['op'] == 'STORE_LOCAL' and ins['arg'] in self.string_params for ins in self.code),
                'refuse writes to immutable STRING parameters')
        if self.helper:
            require(self.result in (INT, BOOL) and
                    all(ins['op'] in OWNED_PRETEST | {'STORE_LOCAL', 'REF_GET', 'RET'} | ({'ADD', 'REF_SET'} if self.exclusive else set()) | ({'SUB', 'ASSERT'} if self.multi else set()) | ({'ADD', 'MUL'} if self.shared_paths else set()) | ({'JMP', 'JMP_TRUE', 'JMP_FALSE'} if self.single_shared_path or self.exclusive_control else set()) for ins in self.code),
                    'require bounded exclusive helper control' if self.exclusive_control else
                    'require total scalar shared-helper diamonds' if self.single_shared_path else
                    'require a straight-line scalar borrowed helper')
        require(0 < len(self.code) <= 4096, 'require bounded function instructions')
        self.path_table = {}
        for instruction in self.code:
            if instruction['op'] in ('BORROW_PATH_SHARED', 'BORROW_PATH_EXCLUSIVE'):
                require(self.paths and len(instruction['args']) == 3,
                        'require exact path borrow operands')
                path_index = instruction['args'][2]
                path = instruction.get('path_fields')
                require(type(path_index) is int and 0 <= path_index < 256 and
                        isinstance(path, list) and 1 <= len(path) <= 2 and
                        all(type(field) is int and 0 <= field <= 65535 for field in path),
                        'require a bounded exact numeric path')
                require(path_index not in self.path_table or self.path_table[path_index] == tuple(path),
                        'require consistent retained path identities')
                self.path_table[path_index] = tuple(path)
        self.positions = {ins['pc']: n for n, ins in enumerate(self.code)}
        require(len(self.positions) == len(self.code), 'require unique instruction boundaries')
        self.positions[self.fn['size']] = len(self.code)
        interiors = {n + offset for n, ins in enumerate(self.code)
                     if ins['op'] == 'REGION_BEGIN' for offset in range(1, self.reference_region_length(n))}
        self.backedges = {}
        for n, ins in enumerate(self.code):
            if ins['op'] in ('JMP', 'JMP_TRUE', 'JMP_FALSE'):
                require(ins.get('target') in self.positions, 'require exact owned branch boundaries')
                target = self.positions[ins['target']]
                require(target not in interiors, 'require branches outside ephemeral read interiors')
                if target <= n:
                    require(ins['op'] == 'JMP' and (self.loop_exits or target not in self.backedges),
                            'require one unconditional backedge per owned loop header')
                    self.backedges[target] = n
        if self.helper and self.single_shared_path:
            require(not self.backedges and self.code[-1]['op'] == 'RET' and
                    sum(ins['op'] == 'RET' for ins in self.code) == 1,
                    'require forward total shared-helper diamonds with one final return')

    def owner_local(self, slot):
        require(slot in self.live, 'require a live exact local owner')
        return self.live[slot]

    def reference_region_length(self, i):
        if self.shared_paths and i + 3 < len(self.code) and self.code[i + 3]['op'] == 'CALL_REF':
            return 5
        if self.multi and i + 4 < len(self.code) and self.code[i + 4]['op'] == 'CALL_REF':
            return 6
        return 4

    def path_place(self, instruction):
        require(len(instruction['args']) == 3, 'require exact path operands')
        slot = instruction['args'][1]
        layout = self.owner_local(slot)
        path = self.path_table[instruction['args'][2]]
        for field in path:
            require(field < len(self.fields[layout]) and isinstance(self.fields[layout][field], tuple),
                    'require an exact owned-child path step')
            layout = self.fields[layout][field][1]
        require(all(tag in (INT, BOOL) for tag in self.fields[layout]),
                'require a scalar-leaf path endpoint')
        return slot, path, layout

    def observation(self, i, end, stack, body, allow_call=True):
        length = self.reference_region_length(i)
        single_path_call = self.single_shared_path or (self.single_exclusive_path and
                           i + 1 < len(self.code) and self.code[i + 1]['op'] == 'BORROW_PATH_EXCLUSIVE')
        if length in (5, 6) or (single_path_call and i + 2 < len(self.code) and
                                  self.code[i + 2]['op'] == 'CALL_REF'):
            count = length - 3
            require(allow_call and self.index == 0 and i + length - 1 < end,
                    'require a complete single-exclusive-path caller region' if self.single_exclusive_path else
                    'require a complete single-shared-path caller region' if self.single_shared_path else
                    'require a complete two-reference caller region' if self.shared_paths else 'require a complete three-reference caller region')
            require(len(self.code[i + 1]['args']) == (3 if self.paths else 2), 'require an exact first reference operand pair')
            base = self.code[i + 1]['args'][0]
            require(count == len(self.module['functions'][1]['params']), 'require exact region parameter count')
            borrows = self.code[i + 1:i + 1 + count]
            call, finish = self.code[i + 1 + count:i + length]
            require(call['args'] == [1, base] and finish['op'] == 'REGION_END' and not finish['args'] and not self.code[i]['args'],
                    'require exact single-exclusive-path call and region end' if self.single_exclusive_path else
                    'require exact single-shared-path call and region end' if self.single_shared_path else
                    'require exact two-reference call and region end' if self.shared_paths else 'require exact three-reference call and region end')
            slots, places = [], []
            for parameter, borrow in enumerate(borrows):
                mode = self.module['functions'][1]['param_modes'][parameter]
                opcode = ('BORROW_PATH_' if self.paths else 'BORROW_LOCAL_') + ('SHARED' if mode == 1 else 'EXCLUSIVE')
                require(borrow['op'] == opcode and len(borrow['args']) == (3 if self.paths else 2) and
                        borrow['args'][0] == base + parameter,
                        'require exact ordered reference modes')
                if self.paths:
                    slot, path, layout = self.path_place(borrow)
                    places.append(path)
                else:
                    slot, layout = borrow['args'][1], self.owner_local(borrow['args'][1])
                require(self.module['functions'][1]['params'][parameter] == [8, layout],
                        'require exact positional borrowed nominal identity')
                slots.append(slot)
            if self.paths:
                require(len(set(slots)) == 1, 'require one live nested root')
                if self.single_shared_path or self.single_exclusive_path:
                    root = (slots[0], self.owner_local(slots[0]))
                    require(self.shared_root is None or self.shared_root == root,
                            'require one exact root slot and layout for exclusive path calls' if self.single_exclusive_path else
                            'require one exact root slot and layout for shared path calls')
                    self.shared_root = root
                require(self.shared_paths or all(left[:min(len(left), len(right))] != right[:min(len(left), len(right))]
                            for n, left in enumerate(places) for right in places[n + 1:]),
                        'require pairwise disjoint nested projections')
            else:
                require(len(set(slots)) == 3, 'require three distinct borrowed roots')
            for offset in range(1, length):
                self.touch(i + offset)
            target = Expr(self.module['functions'][1]['result'], 'temporary', call['pc'])
            body.append(('path_borrow_call', target, slots[0], tuple(places)) if self.paths else
                        ('multi_borrow_call', target, tuple(slots)))
            stack.append(target)
            self.calls.add(1)
            return
        if self.paths and i + 3 < end and self.code[i + 1]['op'] == 'BORROW_PATH_SHARED':
            borrow, read, finish = self.code[i + 1:i + 4]
            require(read['op'] == 'REF_GET' and len(read['args']) == 2 and
                    finish['op'] == 'REGION_END' and not finish['args'] and not self.code[i]['args'] and
                    borrow['args'][0] == read['args'][0], 'require one complete shared path observation')
            slot, path, layout = self.path_place(borrow)
            field = read['args'][1]
            require(0 <= field < len(self.fields[layout]), 'require an exact path scalar field')
            target = Expr(self.fields[layout][field], 'temporary', read['pc'])
            body.append(('path_read', target, slot, path, field))
            stack.append(target)
            for offset in range(1, 4):
                self.touch(i + offset)
            return
        require(i + 3 < end, 'require a complete ephemeral read within its owned region')
        for offset in (1, 2, 3):
            self.touch(i + offset)
        borrow, read, region_end_instruction = self.code[i + 1:i + 4]
        require(borrow['op'] in ('BORROW_LOCAL_SHARED', 'BORROW_LOCAL_EXCLUSIVE') and
                read['op'] in ('REF_GET', 'CALL_REF') and
                region_end_instruction['op'] == 'REGION_END' and len(borrow['args']) == 2 and
                len(read['args']) == 2,
                'require one complete ephemeral reference operation')
        slot = borrow['args'][1]
        layout = self.owner_local(slot)
        if read['op'] == 'CALL_REF':
            require(allow_call, 'refuse calls in owned observation pretests')
            require(borrow['op'] == ('BORROW_LOCAL_EXCLUSIVE' if self.exclusive else 'BORROW_LOCAL_SHARED'),
                    'require the exact helper reference mode')
            require(self.borrowed and self.index == 0 and read['args'] == [1, borrow['args'][0]] and
                    self.module['functions'][1]['params'] == [[8, layout]],
                    'require exact caller owner and borrowed helper identity')
            target = Expr(self.module['functions'][1]['result'], 'temporary', read['pc'])
            body.append(('borrow_call', target, slot))
            self.calls.add(1)
        else:
            require(borrow['op'] == 'BORROW_LOCAL_SHARED', 'require shared ephemeral field reads')
            field = read['args'][1]
            require(borrow['args'][0] == read['args'][0] and
                    0 <= field < len(self.fields[layout]),
                    'require a field within the exact owner layout')
            require(all(t in (INT, BOOL) for t in self.fields[layout]),
                    'require a scalar-leaf owner for ephemeral field observation')
            require(self.fields[layout][field] in (INT, BOOL), 'require a scalar shared field')
            target = Expr(self.fields[layout][field], 'temporary', read['pc'])
            body.append(('own_read', target, slot, field))
        stack.append(target)

    def helper_read(self, ins, stack, body):
        arg = ins['arg']
        require(self.helper and len(ins['args']) == 2 and 0 <= arg < len(self.params),
                'require the exact borrowed helper parameter reference')
        field, layout = ins['args'][1], self.params[arg][1]
        require(0 <= field < len(self.fields[layout]), 'require an exact borrowed parameter field')
        require(self.fields[layout][field] in (INT, BOOL), 'require a scalar shared field')
        target = Expr(self.fields[layout][field], 'temporary', ins['pc'])
        body.append(('multi_borrow_read', target, arg, field) if self.multi else ('borrow_read', target, field))
        stack.append(target)

    def region(self, start, end, initialized, live, used, depth=0):
        require(depth <= 32, 'limit owned region nesting to 32')
        initialized = set(initialized)
        self.live, self.used = dict(live), set(used)
        stack = []
        body = []
        i = start
        while i < end:
            if i in self.backedges:
                require(not stack, 'require an empty owned loop-entry stack')
                back = self.backedges[i]
                require(i < back < end, 'require an owned loop within its enclosing region')
                branch = next((n for n in range(i, back)
                               if self.code[n]['op'] in ('JMP_TRUE', 'JMP_FALSE')), None)
                require(branch is not None, 'require an owned loop pretest')
                require(self.positions[self.code[branch]['target']] == back + 1,
                        'require an owned pretest exit immediately after its backedge')
                condition_stack, condition_statements = [], []
                materialized = any(self.code[n]['op'] == 'REGION_BEGIN' or
                                   self.code[n]['op'] in OWNED_WRAPPING_PRETEST or
                                   (self.helper and self.exclusive_control and self.code[n]['op'] == 'REF_GET')
                                   for n in range(i, branch))
                n, pretest_calls = i, 0
                while n < branch:
                    condition_instruction = self.code[n]
                    if materialized and condition_instruction['op'] == 'REGION_BEGIN':
                        self.touch(n)
                        length = self.reference_region_length(n)
                        is_call = any(ins['op'] == 'CALL_REF' for ins in self.code[n:n + length])
                        if is_call:
                            require(self.index == 0 and pretest_calls == 0 and
                                    ((self.paths and not self.shared_paths and length == 6 and
                                      self.module['functions'][1]['param_modes'] == [2, 2, 1]) or
                                     (self.single_exclusive_path and length == 4)),
                                    'require one exact exclusive path call per owned pretest' if self.single_exclusive_path else
                                    'require one exact three-path reference call per owned pretest')
                            pretest_calls += 1
                        self.observation(n, branch, condition_stack, condition_statements, allow_call=is_call)
                        n += length
                        continue
                    if self.helper and self.exclusive_control and condition_instruction['op'] == 'REF_GET':
                        self.touch(n)
                        self.helper_read(condition_instruction, condition_stack, condition_statements)
                        n += 1
                        continue
                    require(condition_instruction['op'] in OWNED_PRETEST | OWNED_WRAPPING_PRETEST,
                            'require a nontrapping pure scalar owned pretest')
                    if condition_instruction['op'] == 'LOAD_LOCAL':
                        require(condition_instruction['arg'] in self.types,
                                'require a scalar owned pretest local')
                    self.simple(n, condition_stack, initialized, condition_statements, pure=not materialized)
                    require(all(value.tag in (INT, BOOL) for value in condition_stack),
                            'require exact scalar owned pretest values')
                    n += 1
                condition = self.pop(condition_stack, BOOL)
                require(not condition_stack, 'require one owned loop condition')
                if self.code[branch]['op'] == 'JMP_TRUE':
                    condition = Expr(BOOL, 'not', None, (condition,))
                self.touch(branch)
                entry_live, entry_used = dict(self.live), set(self.used)
                loop_body, _, loop_live, loop_used, loop_returns = self.region(
                    branch + 1, back, initialized, entry_live, entry_used, depth + 1)
                require(loop_returns or loop_live == entry_live,
                        'require exact carried owner identities at the backedge')
                self.touch(back)
                body.append(('while_observe', condition, condition_statements, loop_body) if materialized else
                            ('while', condition, loop_body))
                # I retain the zero-iteration state. Body-local declarations execute
                # inside while on each iteration; I analyze their syntax only once.
                self.live, self.used = entry_live, entry_used | loop_used
                i = back + 1
                continue
            ins = self.code[i]
            op, arg, pc = ins['op'], ins['arg'], ins['pc']
            if op in ('JMP_TRUE', 'JMP_FALSE'):
                self.touch(i)
                condition = self.pop(stack, BOOL)
                require(not stack, 'require empty operand stacks at owned branches')
                target = self.positions[ins['target']]
                require(i < target <= end, 'require a forward owned region boundary')
                if op == 'JMP_TRUE':
                    condition = Expr(BOOL, 'not', None, (condition,))
                separator = target - 1
                has_else = separator > i and self.code[separator]['op'] == 'JMP'
                join = self.positions[self.code[separator]['target']] if has_else else target
                require(target <= join <= end, 'require a non-crossing owned diamond')
                entry_live, entry_used = dict(self.live), set(self.used)
                left, li, ll, lu, lr = self.region(i + 1, separator if has_else else target,
                                                  initialized, entry_live, entry_used, depth + 1)
                if has_else:
                    self.touch(separator)
                    right, ri, rl, ru, rr = self.region(target, join, initialized,
                                                       entry_live, entry_used, depth + 1)
                else:
                    right, ri, rl, ru, rr = [], set(initialized), entry_live, entry_used, False
                body.append(('if', condition, left, right))
                self.used = lu | ru
                if lr and rr:
                    require(join == end, 'do not discard a terminal owned continuation')
                    return body, initialized, {}, self.used, True
                if lr:
                    initialized, self.live = ri, rl
                elif rr:
                    initialized, self.live = li, ll
                else:
                    require(ll == rl, 'require exact live-owner agreement at joins')
                    initialized, self.live = li & ri, ll
                require(all(entry_live.get(slot) == layout for slot, layout in self.live.items()),
                        'refuse branch-local owners escaping a structured region')
                i = join
                continue
            require(op != 'JMP', 'refuse unstructured owned jumps')
            if op not in SIMPLE or op == 'CALL':
                self.touch(i)
            if op == 'PUSH_STR':
                require(not self.borrowed and arg in self.string_literals, 'require a retained executable string literal')
                stack.append(Expr(STRING, 'string_literal', (arg, self.string_literals[arg])))
            elif op == 'LOAD_LOCAL' and arg in self.string_params:
                self.touch(i)
                require(arg in initialized, 'require an initialized immutable string parameter')
                stack.append(Expr(STRING, 'string_parameter', arg))
            elif op in ('PRINT', 'PRINTLN'):
                require(not self.borrowed and ins['args'] == [], 'require an exact consuming string print')
                body.append(('print_string', self.pop(stack, STRING), op == 'PRINTLN'))
            elif op == 'OWN_PACK':
                require(0 <= arg < len(self.layouts), 'require a retained nominal layout')
                fields = self.fields[arg]
                values = tuple(reversed([self.pop(stack, t) for t in reversed(fields)]))
                target = Expr(owned_type(arg), 'temporary', pc)
                body.append(('own_pack', target, arg, values))
                stack.append(target)
            elif op == 'OWN_STORE_LOCAL':
                require(0 <= arg < len(self.declarations) and arg not in self.live,
                        'require a consumed declared owner destination')
                repeated = arg in self.used
                require(not repeated or (self.index == 0 and self.borrowed and
                        self.module['functions'][1].get('param_modes') == [2]),
                        'require single-exclusive entry owner reassignment')
                tag, layout = self.declarations[arg]
                require(tag == 8, 'require an owned record local')
                value = self.pop(stack, owned_type(layout))
                body.append(('own_assign' if repeated else 'own_store', arg, layout, value))
                self.live[arg] = layout
                self.used.add(arg)
            elif op == 'OWN_MOVE_LOCAL':
                layout = self.owner_local(arg)
                target = Expr(owned_type(layout), 'temporary', pc)
                body.append(('own_move', target, arg))
                del self.live[arg]
                stack.append(target)
            elif op == 'OWN_UNPACK_LOCAL':
                layout = self.owner_local(arg)
                values = tuple(Expr(t, 'temporary', f'{pc}_{f}')
                               for f, t in enumerate(self.fields[layout]))
                body.append(('own_unpack', arg, layout, values))
                del self.live[arg]
                stack.extend(values)
            elif op == 'REF_SET':
                require(self.helper and self.exclusive and len(ins['args']) == 2 and 0 <= arg < len(self.params) and self.modes[arg] == 2,
                        'require the exact exclusive helper reference')
                field, layout = ins['args'][1], self.params[arg][1]
                require(0 <= field < len(self.fields[layout]), 'require an exact exclusive field')
                value = self.pop(stack, self.fields[layout][field])
                body.append(('multi_borrow_write', arg, field, value) if self.multi else ('borrow_write', field, value))
            elif op == 'REF_GET':
                self.helper_read(ins, stack, body)
            elif op == 'REGION_BEGIN':
                self.observation(i, end, stack, body)
                i += self.reference_region_length(i) - 1
            elif op == 'CALL':
                require(not self.borrowed, 'refuse mixed consuming and borrowed calls')
                require(0 < arg < len(self.module['functions']), 'require a direct non-entry callee')
                callee = self.module['functions'][arg]
                params = [owned_type(t[1]) if t[0] == 8 else t[0] for t in callee['params']]
                values = tuple(reversed([self.pop(stack, t) for t in reversed(params)]))
                tag, layout = callee['result_type']
                result_type = owned_type(layout) if tag == 8 else tag
                call = Expr(result_type, 'call', arg, values)
                if result_type == VOID:
                    body.append(('call_void', call))
                else:
                    target = Expr(result_type, 'temporary', pc)
                    body.append(('let', target, call))
                    stack.append(target)
                self.calls.add(arg)
            elif op == 'ASSERT':
                body.append(('assert', self.pop(stack, BOOL)))
            elif op == 'RET':
                require(i == end - 1 and not self.live,
                        'require all owners consumed before terminal return')
                if self.result == VOID:
                    require(not stack, 'require an empty operand stack at VOID return')
                    body.append(('return_void',))
                else:
                    result = self.pop(stack, self.result)
                    require(not stack, 'require an empty operand stack after exact return')
                    body.append(('return', result))
                return body, initialized, self.live, self.used, True
            else:
                require(op in SIMPLE and op not in ('CALL', 'PUSH_U8', 'PUSH_F64', 'F64_FROM_BITS', 'F64_TO_BITS', 'F64_NEG'),
                        'refuse unsupported resource-route instruction ' + op)
                # I isolate the suffix above the last staged owner before scalar evaluation.
                split = max((n + 1 for n, value in enumerate(stack)
                             if value.tag not in (INT, BOOL)), default=0)
                prefix, scalar = stack[:split], stack[split:]
                if op in ('LOAD_LOCAL', 'STORE_LOCAL'):
                    require(arg in self.types, 'require a declared scalar local')
                self.simple(i, scalar, initialized, body)
                require(all(value.tag in (INT, BOOL) for value in scalar),
                        'require exact scalar resource-route values')
                stack = prefix + scalar
            i += 1
        require(not stack, 'require an empty operand stack at owned region exit')
        return body, initialized, self.live, self.used, False

    def exit_step(self, i, end, stack, initialized, body):
        ins = self.code[i]
        op, arg, pc = ins['op'], ins['arg'], ins['pc']
        if op not in SIMPLE or op == 'CALL':
            self.touch(i)
        if op == 'OWN_PACK':
            require(0 <= arg < len(self.layouts), 'require a retained nominal layout')
            fields = self.fields[arg]
            values = tuple(reversed([self.pop(stack, t) for t in reversed(fields)]))
            target = Expr(owned_type(arg), 'temporary', pc)
            body.append(('own_pack', target, arg, values))
            stack.append(target)
        elif op == 'OWN_STORE_LOCAL':
            require(0 <= arg < len(self.declarations) and arg not in self.live,
                    'require a consumed declared owner destination')
            repeated = arg in self.used
            require(not repeated or (self.index == 0 and self.borrowed and
                    self.module['functions'][1].get('param_modes') == [2]),
                    'require single-exclusive entry owner reassignment')
            tag, layout = self.declarations[arg]
            require(tag == 8, 'require an owned record local')
            value = self.pop(stack, owned_type(layout))
            body.append(('own_assign' if repeated else 'own_store', arg, layout, value))
            self.live[arg] = layout
            self.used.add(arg)
        elif op == 'OWN_MOVE_LOCAL':
            layout = self.owner_local(arg)
            target = Expr(owned_type(layout), 'temporary', pc)
            body.append(('own_move', target, arg))
            del self.live[arg]
            stack.append(target)
        elif op == 'OWN_UNPACK_LOCAL':
            layout = self.owner_local(arg)
            values = tuple(Expr(t, 'temporary', f'{pc}_{f}')
                           for f, t in enumerate(self.fields[layout]))
            body.append(('own_unpack', arg, layout, values))
            del self.live[arg]
            stack.extend(values)
        elif op == 'REF_SET':
            require(self.helper and self.exclusive and len(ins['args']) == 2 and 0 <= arg < len(self.params) and self.modes[arg] == 2,
                    'require the exact exclusive helper reference')
            field, layout = ins['args'][1], self.params[arg][1]
            require(0 <= field < len(self.fields[layout]), 'require an exact exclusive field')
            value = self.pop(stack, self.fields[layout][field])
            body.append(('multi_borrow_write', arg, field, value) if self.multi else ('borrow_write', field, value))
        elif op == 'REF_GET':
            self.helper_read(ins, stack, body)
        elif op == 'REGION_BEGIN':
            self.observation(i, end, stack, body)
            i += self.reference_region_length(i) - 1
        elif op == 'CALL':
            require(not self.borrowed, 'refuse mixed consuming and borrowed calls')
            require(0 < arg < len(self.module['functions']), 'require a direct non-entry callee')
            callee = self.module['functions'][arg]
            params = [owned_type(t[1]) if t[0] == 8 else t[0] for t in callee['params']]
            values = tuple(reversed([self.pop(stack, t) for t in reversed(params)]))
            tag, layout = callee['result_type']
            result_type = owned_type(layout) if tag == 8 else tag
            call = Expr(result_type, 'call', arg, values)
            if result_type == VOID:
                body.append(('call_void', call))
            else:
                target = Expr(result_type, 'temporary', pc)
                body.append(('let', target, call))
                stack.append(target)
            self.calls.add(arg)
        elif op == 'ASSERT':
            body.append(('assert', self.pop(stack, BOOL)))
        else:
            require(op in SIMPLE and op not in ('CALL', 'PUSH_U8', 'PUSH_F64', 'F64_FROM_BITS', 'F64_TO_BITS', 'F64_NEG'),
                    'refuse unsupported resource-route instruction ' + op)
            # I isolate the suffix above the last staged owner before scalar evaluation.
            split = max((n + 1 for n, value in enumerate(stack)
                         if value.tag not in (INT, BOOL)), default=0)
            prefix, scalar = stack[:split], stack[split:]
            if op in ('LOAD_LOCAL', 'STORE_LOCAL'):
                require(arg in self.types, 'require a declared scalar local')
            self.simple(i, scalar, initialized, body)
            require(all(value.tag in (INT, BOOL) for value in scalar),
                    'require exact scalar resource-route values')
            stack[:] = prefix + scalar
        return i + 1

    def run(self):
        if self.loop_exits:
            from nanoisa_owned_exits import ExitRegions
            return ExitRegions(self).run()
        parameters = [('own_parameter', i, layout) for i, layout in self.live.items()]
        body, _, _, _, returns = self.region(0, len(self.code), set(range(len(self.params))),
                                             self.live, self.used)
        require(returns, 'require every owned function path to return')
        if (self.single_shared_path or self.single_exclusive_path) and self.index == 0:
            require(self.shared_root is not None, 'require an actual single-exclusive-path caller region' if self.single_exclusive_path else
                    'require an actual single-shared-path caller region')
        require(len(self.seen) == len(self.code), 'require complete owned instruction coverage')
        return Function(self.fn['name'], self.params, self.result, self.types, self.fn['names'], parameters + body)


class OwnedEmit(Emit):
    def __init__(self, functions, language, layouts):
        super().__init__(functions, language)
        self.layouts = layouts
        self.exclusive = any(isinstance(t, tuple) and t[0] == 'borrow' and len(t) == 3 and t[2] == 2
                             for fn in functions for t in fn.params)
        modes = [t[2] if isinstance(t, tuple) and t[0] == 'borrow' else 0 for t in functions[1].params] if len(functions) == 2 else []
        self.loop_exits = any(getattr(fn, 'loop_exits', False) for fn in functions)
        if self.loop_exits:
            require(len(functions) == 2 and modes == [2] and all(getattr(fn, 'loop_exits', False) for fn in functions),
                    'require one complete exit-aware single-exclusive graph')
            from nanoisa_owned_exits import prove_exit_origins
            prove_exit_origins(functions[0])
        self.reassigned = frozenset()
        if any(stmt[0] == 'own_assign' for function in functions for stmt in self.walk_statements(function.body)):
            require(modes == [2] and not any(stmt[0] == 'own_assign' for stmt in self.walk_statements(functions[1].body)),
                    'require single-exclusive entry reassignment origins')
            from nanoisa_owned_origins import prove_owner_reassignment
            self.reassigned = prove_owner_reassignment(functions[0])
        self.shared_paths = modes == [1, 1]
        self.single_shared_path = modes == [1] and any(
            stmt[0] == 'path_borrow_call' for stmt in self.walk_statements(functions[0].body))
        self.single_exclusive_path = modes == [2] and any(
            stmt[0] == 'path_borrow_call' for stmt in self.walk_statements(functions[0].body))
        self.exclusive_control = modes == [2] and (self.loop_exits or self.single_exclusive_path or any(
            stmt[0] in ('if', 'while', 'while_observe') for stmt in self.walk_statements(functions[1].body)))
        self.control_proof = None
        if self.exclusive_control:
            from nanoisa_owned_control import prove_exclusive_helper_control
            if self.loop_exits:
                from nanoisa_owned_exits import prove_exit_helper
                self.control_proof = prove_exit_helper(functions[1])
            else:
                self.control_proof = prove_exclusive_helper_control(functions[1])
        self.paths = modes in ([2, 2, 1], [1, 1]) or self.single_shared_path or self.single_exclusive_path
        self.multi = modes in ([2, 1, 2], [2, 2, 1], [1, 1])
        self.witness = (self.exclusive_control_tree(functions, layouts) if self.exclusive_control else
                        self.total_shared_tree(functions, layouts) if self.single_shared_path else
                        self.path_witness(functions, layouts) if self.paths else
                        self.multi_witness(functions) if self.multi else None)
        if (self.paths or self.exclusive_control) and language == 'nano':
            from copy import deepcopy
            self.functions = deepcopy(functions)
            for function in self.functions:
                if self.paths:
                    function.body = self.fold_adjacent_assertions(function.body)
                if self.exclusive_control:
                    function.body = self.fold_add_snapshots(function.body)
        self.inline_add = (self.exclusive or self.shared_paths or self.single_shared_path) and language == 'nano'
        self.inline_mul = self.shared_paths and language == 'nano'
        self.inline_sub = (self.multi or self.loop_exits) and language == 'nano'
        self.inline_div = self.loop_exits and language == 'nano'
        if self.inline_add:
            def validate_inline(node, direct=False):
                if isinstance(node, Expr):
                    require(node.kind != 'arithmetic' or node.value not in (({'add', 'sub'} if self.inline_sub else {'add'}) | ({'mul'} if self.inline_mul else set()) | ({'div'} if self.inline_div else set())) or direct,
                            'require guarded arithmetic at an explicit inline statement boundary')
                    for child in node.args:
                        validate_inline(child)
                elif isinstance(node, (list, tuple)):
                    if node and node[0] == 'let':
                        validate_inline(node[1])
                        validate_inline(node[2], True)
                    else:
                        for child in node:
                            validate_inline(child)
            for function in self.functions:
                validate_inline(function.body)
        self.fields = [[owned_type(child) if tag == 8 else tag
                        for tag, child in layout.get('field_types', [[t, 4294967295] for t in layout['fields']])]
                       for layout in layouts]
        self.reassignment_plans = []
        self.scratch_functions = set()
        if (self.reassigned or self.loop_exits) and language == 'nano':
            from nanoisa_owned_rendering import plan_reassignment_nano
            planned = []
            for function in self.functions:
                lowered, receipt = plan_reassignment_nano(function)
                planned.append(lowered)
                self.reassignment_plans.append(receipt)
                if any(stmt[0] == 'let' and stmt[2].kind == 'arithmetic' and stmt[2].value == 'add'
                       for stmt in self.walk_statements(lowered.body)):
                    self.scratch_functions.add(id(lowered))
            self.functions = planned

    @staticmethod
    def walk_statements(body):
        for stmt in body:
            yield stmt
            if stmt[0] == 'if':
                yield from OwnedEmit.walk_statements(stmt[2])
                yield from OwnedEmit.walk_statements(stmt[3])
            elif stmt[0] == 'while':
                yield from OwnedEmit.walk_statements(stmt[2])
            elif stmt[0] == 'while_observe':
                yield from OwnedEmit.walk_statements(stmt[2])
                yield from OwnedEmit.walk_statements(stmt[3])

    @staticmethod
    def total_shared_tree(functions, layouts):
        # I synthesize only a consistency fixture for the separately total helper.
        def total(node):
            if isinstance(node, Expr):
                return (node.tag in (INT, BOOL) and node.kind in
                        ('constant', 'local', 'temporary', 'binary', 'not', 'bool_int', 'unsigned_compare') and
                        (node.kind != 'binary' or node.value in ('==', '!=', '<', '<=', '>', '>=', 'and', 'or')) and
                        all(total(child) for child in node.args))
            if isinstance(node, (list, tuple)):
                return all(total(child) for child in node)
            return True
        require(all(stmt[0] in ('let', 'store', 'borrow_read', 'if', 'return')
                    for stmt in OwnedEmit.walk_statements(functions[1].body)) and total(functions[1].body),
                'require a total observation-only helper for the synthesized tree fixture')
        return OwnedEmit.zero_path_tree(functions, layouts)

    @staticmethod
    def zero_path_tree(functions, layouts):
        statements = list(OwnedEmit.walk_statements(functions[0].body))
        calls = [stmt for stmt in statements if stmt[0] == 'path_borrow_call']
        require(calls and all(stmt[2] == calls[0][2] and len(stmt[3]) == 1 for stmt in calls),
                'require one shared root and exact single path calls')
        roots = [stmt[2] for stmt in statements if stmt[0] == 'own_store' and stmt[1] == calls[0][2]]
        require(len(roots) == 1, 'require one exact shared root declaration')
        def zero_tree(layout):
            return (layout, tuple(zero_tree(child) if tag == 8 else
                    Expr(tag, 'constant', 0 if tag == INT else False)
                    for tag, child in layouts[layout]['field_types']))
        return zero_tree(roots[0]), calls[0][3], ()

    @staticmethod
    def exclusive_control_tree(functions, layouts):
        if any(stmt[0] == 'path_borrow_call' for stmt in OwnedEmit.walk_statements(functions[0].body)):
            return OwnedEmit.zero_path_tree(functions, layouts)
        layout = functions[1].params[0][1]
        fields = layouts[layout]['field_types']
        require(all(tag in (INT, BOOL) for tag, _ in fields), 'require a scalar exclusive shadow leaf')
        return (layout, tuple(Expr(tag, 'constant', 0 if tag == INT else False)
                              for tag, _ in fields)), None, ()

    @staticmethod
    def fold_add_snapshots(body):
        # I move only a sole-use scalar local read into the immediately following
        # guarded ADD operand snapshot. I retain its result let and every store.
        from collections import Counter
        occurrences = Counter()
        def count(node):
            if isinstance(node, Expr):
                occurrences[node] += 1
                for arg in node.args:
                    count(arg)
            elif isinstance(node, (list, tuple)):
                for arg in node:
                    count(arg)
        count(body)
        def rewrite(statements):
            result, index = [], 0
            while index < len(statements):
                stmt = statements[index]
                if stmt[0] == 'let' and stmt[1].tag == INT and stmt[1].kind == 'temporary' and \
                        stmt[2].tag == INT and stmt[2].kind == 'local' and not stmt[2].args and \
                        occurrences[stmt[1]] == 2 and index + 1 < len(statements):
                    following = statements[index + 1]
                    if following[0] == 'let':
                        value = following[2]
                        if value.tag == INT and value.kind == 'arithmetic' and value.value == 'add' and \
                                len(value.args) == 2 and stmt[1] in value.args and all(
                                    arg.tag == INT and arg.kind in ('constant', 'local', 'temporary') and not arg.args
                                    for arg in value.args):
                            args = tuple(stmt[2] if arg == stmt[1] else arg for arg in value.args)
                            result.append(('let', following[1], Expr(INT, 'arithmetic', 'add', args)))
                            index += 2
                            continue
                if stmt[0] == 'if':
                    stmt = (stmt[0], stmt[1], rewrite(stmt[2]), rewrite(stmt[3]))
                elif stmt[0] == 'while':
                    stmt = (stmt[0], stmt[1], rewrite(stmt[2]))
                elif stmt[0] == 'while_observe':
                    stmt = (stmt[0], stmt[1], rewrite(stmt[2]), rewrite(stmt[3]))
                result.append(stmt)
                index += 1
            return result
        return rewrite(body)

    @staticmethod
    def fold_adjacent_assertions(body):
        # I count declarations and uses before one pass; shared snapshots stay explicit.
        from collections import Counter
        occurrences = Counter()
        def count(node):
            if isinstance(node, Expr):
                occurrences[node] += 1
                for child in node.args:
                    count(child)
            elif isinstance(node, (list, tuple)):
                for child in node:
                    count(child)
        count(body)
        def total(expr):
            if not isinstance(expr, Expr) or expr.tag not in (INT, BOOL):
                return False
            if expr.kind in ('constant', 'local', 'temporary'):
                return not expr.args
            if expr.kind == 'not':
                return expr.tag == BOOL and len(expr.args) == 1 and expr.args[0].tag == BOOL and total(expr.args[0])
            if expr.kind == 'binary' and expr.value in ('==', '!=', '<', '<=', '>', '>='):
                return (expr.tag == BOOL and len(expr.args) == 2 and expr.args[0].tag == expr.args[1].tag and
                        all(total(child) for child in expr.args))
            return False
        def fold(statements):
            result, i = [], 0
            while i < len(statements):
                stmt = statements[i]
                if (stmt[0] == 'let' and isinstance(stmt[1], Expr) and stmt[1].kind == 'temporary' and
                    stmt[1].tag == BOOL and stmt[2].tag == BOOL and total(stmt[2]) and occurrences[stmt[1]] == 2 and
                    i + 1 < len(statements) and statements[i + 1] == ('assert', stmt[1])):
                    result.append(('assert', stmt[2]))
                    i += 2
                    continue
                if stmt[0] == 'if':
                    stmt = (stmt[0], stmt[1], fold(stmt[2]), fold(stmt[3]))
                elif stmt[0] == 'while':
                    stmt = (stmt[0], stmt[1], fold(stmt[2]))
                elif stmt[0] == 'while_observe':
                    stmt = (stmt[0], stmt[1], fold(stmt[2]), fold(stmt[3]))
                result.append(stmt)
                i += 1
            return result
        return fold(body)

    @staticmethod
    def multi_witness(functions):
        # I retain literal snapshots, never evaluate the reconstructed program.
        values, owners = {}, {}
        def literal(expr):
            return expr if expr.kind == 'constant' and expr.tag in (INT, BOOL) else values.get(expr)
        witness = None
        for stmt in functions[0].body:
            kind = stmt[0]
            if kind == 'let':
                values[stmt[1]] = literal(stmt[2])
            elif kind == 'store':
                values[Expr(stmt[2].tag, 'local', stmt[1])] = literal(stmt[2])
            elif kind == 'own_pack':
                vector = tuple(literal(value) for value in stmt[3])
                owners[stmt[1]] = vector if all(v is not None for v in vector) else None
            elif kind == 'own_store':
                owners[stmt[1]] = owners.pop(stmt[3], None)
            elif kind == 'own_move':
                owners[stmt[1]] = owners.pop(stmt[2], None)
            elif kind == 'multi_borrow_call':
                witness = tuple(owners.get(slot) for slot in stmt[2])
                require(all(vector is not None for vector in witness),
                        'require actual first-call literal owner snapshots')
                break
            else:
                require(False, 'require a straight-line literal witness prefix')
        require(witness is not None, 'require an actual direct first-call witness')
        provenance = {}
        for stmt in functions[1].body:
            kind = stmt[0]
            if kind == 'multi_borrow_read':
                provenance[stmt[1]] = (stmt[2], stmt[3]) if stmt[2] == 1 and stmt[1].tag == BOOL else None
            elif kind == 'let':
                provenance[stmt[1]] = provenance.get(stmt[2])
            elif kind == 'store':
                provenance[Expr(stmt[2].tag, 'local', stmt[1])] = provenance.get(stmt[2])
            elif kind == 'assert':
                source = provenance.get(stmt[1])
                require(source is not None and witness[source[0]][source[1]].tag == BOOL and
                        witness[source[0]][source[1]].value is True,
                        'require a true actual shared-BOOL assertion witness')
            else:
                require(kind in ('multi_borrow_write', 'return'),
                        'require total straight-line witness helper statements')
        return witness

    @staticmethod
    def path_witness(functions, layouts):
        values, owners = {}, {}
        def literal(expr):
            return expr if expr.kind == 'constant' and expr.tag in (INT, BOOL) else values.get(expr)
        witness = None
        for stmt in functions[0].body:
            kind = stmt[0]
            if kind == 'let':
                values[stmt[1]] = literal(stmt[2])
            elif kind == 'store':
                values[Expr(stmt[2].tag, 'local', stmt[1])] = literal(stmt[2])
            elif kind == 'own_pack':
                vector = []
                for descriptor, operand in zip(layouts[stmt[2]]['field_types'], stmt[3]):
                    if descriptor[0] == 8:
                        child = owners.pop(operand, None)
                        require(child is not None and child[0] == descriptor[1],
                                'require literal exact child snapshots')
                        vector.append(child)
                    else:
                        vector.append(literal(operand))
                owners[stmt[1]] = (stmt[2], tuple(vector)) if all(v is not None for v in vector) else None
            elif kind == 'own_store':
                owners[stmt[1]] = owners.pop(stmt[3], None)
            elif kind == 'own_move':
                owners[stmt[1]] = owners.pop(stmt[2], None)
            elif kind == 'path_borrow_call':
                tree = owners.get(stmt[2])
                require(tree is not None, 'require the actual first-call literal tree snapshot')
                leaves = []
                for parameter, path in enumerate(stmt[3]):
                    node = tree
                    for field in path:
                        require(isinstance(node, tuple) and field < len(node[1]), 'require a literal path witness')
                        node = node[1][field]
                    require(isinstance(node, tuple) and node[0] == functions[1].params[parameter][1],
                            'require the exact witness leaf identity')
                    leaves.append(node[1])
                witness = (tree, stmt[3], tuple(leaves))
                break
            else:
                require(False, 'require a straight-line literal tree witness prefix')
        require(witness is not None, 'require an actual direct tree-call witness')
        provenance = {}
        for stmt in functions[1].body:
            kind = stmt[0]
            if kind == 'multi_borrow_read':
                provenance[stmt[1]] = (stmt[2], stmt[3]) if functions[1].params[stmt[2]][2] == 1 and stmt[1].tag == BOOL else None
            elif kind == 'let':
                provenance[stmt[1]] = provenance.get(stmt[2])
            elif kind == 'store':
                provenance[Expr(stmt[2].tag, 'local', stmt[1])] = provenance.get(stmt[2])
            elif kind == 'assert':
                source = provenance.get(stmt[1])
                require(source is not None and witness[2][source[0]][source[1]].tag == BOOL and witness[2][source[0]][source[1]].value is True,
                        'require a true actual tree shared-BOOL assertion witness')
            else:
                require(kind in ('multi_borrow_write', 'return'), 'require total straight-line tree helper statements')
        return witness

    def path_shadow_function(self):
        tree, paths, _ = self.witness
        body, captures, results = [], [], []
        slot_count, temporary_count = 0, 0
        def temporary(tag):
            nonlocal temporary_count
            temporary_count += 1
            return Expr(tag, 'temporary', f'shadow_{temporary_count}')
        def store(value, layout):
            nonlocal slot_count
            slot = slot_count
            slot_count += 1
            body.append(('own_store', slot, layout, value))
            return slot
        def pack(node):
            layout, vector = node
            operands = tuple(pack(child) if isinstance(child, tuple) else child for child in vector)
            target = temporary(owned_type(layout))
            body.append(('own_pack', target, layout, operands))
            return target
        def consume(slot, node, output):
            layout, vector = node
            targets = tuple(temporary(owned_type(v[0]) if isinstance(v, tuple) else v.tag) for v in vector)
            body.append(('own_unpack', slot, layout, targets))
            for value, target in zip(vector, targets):
                if isinstance(value, tuple):
                    consume(store(target, value[0]), value, output)
                else:
                    output.append(target)
        for _ in range(2):
            root = store(pack(tree), tree[0])
            result = temporary(self.functions[1].result)
            body.append(('borrow_call', result, root) if paths is None else
                        ('path_borrow_call', result, root, paths))
            results.append(result)
            fields = []
            consume(root, tree, fields)
            if self.single_shared_path:
                for field in fields:
                    value = Expr(field.tag, 'constant', 0 if field.tag == INT else False)
                    body.append(('assert', Expr(BOOL, 'binary', '==', (field, value))))
            captures.append(fields)
        body.append(('assert', Expr(BOOL, 'binary', '==', tuple(results))))
        for left, right in zip(*captures):
            body.append(('assert', Expr(BOOL, 'binary', '==', (left, right))))
        return Function('tree_shadow', [], VOID, {}, [''] * slot_count, body)

    def path_shadow(self):
        function = self.path_shadow_function()
        require(self.nano_local_cost(function) <= 128, 'limit generated tree shadow to 128 local slots')
        if self.exclusive_control:
            # I count every emitted tree statement and operand conservatively,
            # then include both independently proved helper invocations.
            def work(node):
                if isinstance(node, Expr):
                    return 1 + sum(work(arg) for arg in node.args)
                if isinstance(node, (list, tuple)):
                    return 1 + sum(work(arg) for arg in node)
                return 1
            self.control_proof['shadow_work'] = 2 * self.control_proof['work'] + 4 * work(function.body)
        saved_lines, saved_function = self.lines, self.function
        self.lines, self.function = [], function
        try:
            self.statements(function.body)
            return f'\nshadow {self.name(1)} {{\n' + '\n'.join(self.lines) + '\n}\n'
        finally:
            self.lines, self.function = saved_lines, saved_function

    def nano_local_cost(self, function):
        # My emitted subset allocates named lets, constructor staging and pattern slots.
        def statements_cost(body):
            total = 0
            for stmt in body:
                kind = stmt[0]
                if kind == 'let':
                    value = stmt[2]
                    if self.inline_div and value.kind == 'arithmetic' and value.value == 'div':
                        total += 4
                        continue
                    if self.inline_mul and value.kind == 'arithmetic' and value.value == 'mul':
                        total += 27
                        continue
                    total += (1 if id(function) in self.scratch_functions and value.kind == 'arithmetic' and value.value == 'add' else
                              7 if value.kind == 'arithmetic' and value.value in ({'add', 'sub'} if self.inline_sub else {'add'}) else 1)
                elif kind in ('borrow_read', 'borrow_call', 'multi_borrow_read', 'multi_borrow_call', 'path_read', 'path_borrow_call', 'own_store', 'own_move', 'own_read', 'own_parameter'):
                    total += 1
                elif kind == 'own_pack':
                    total += 1 + len(stmt[3])
                elif kind == 'own_pack_store':
                    total += 1 + len(stmt[3])
                elif kind == 'own_move_store':
                    total += 0 if stmt[4] else 1
                elif kind == 'own_read_store':
                    pass
                elif kind in ('own_unpack', 'own_unpack_pruned'):
                    # Child-bearing patterns stage every field before named projections.
                    staged = any(isinstance(target.tag, tuple) for target in stmt[3])
                    total += 1 + (2 if staged else 1) * len(stmt[3]) + (sum(stmt[4]) if kind == 'own_unpack_pruned' else len(stmt[3]))
                elif kind == 'if':
                    total += statements_cost(stmt[2]) + statements_cost(stmt[3])
                elif kind == 'while':
                    total += statements_cost(stmt[2])
                elif kind == 'while_observe':
                    total += statements_cost(stmt[2]) + statements_cost(stmt[3])
                else:
                    require(kind in ('store', 'own_assign', 'borrow_write', 'multi_borrow_write', 'assert', 'return', 'break', 'continue', 'call_void', 'return_void', 'print_string'),
                            'require counted exclusive Nano statement storage')
            return total
        return len(function.locals) + len(function.params) + statements_cost(function.body) + (5 if id(function) in self.scratch_functions else 0)

    def name(self, index):
        return 'main' if self.language == 'nano' and index == 0 else super().name(index)

    def expression(self, expr):
        if expr.kind == 'string_parameter':
            return f'nlr_a{expr.value}'
        if expr.kind == 'string_literal':
            index, data = expr.value
            return (f'(NlrString){{nlr_string_{index}, {len(data)}}}' if self.language == 'c' else nano_literal(data))
        return super().expression(expr)

    def type(self, tag):
        if tag == STRING:
            return 'NlrString' if self.language == 'c' else 'string'
        if tag == VOID:
            return 'void'
        if isinstance(tag, tuple) and tag[0] == 'borrow':
            mutable = len(tag) == 3 and tag[2] == 2
            return (('' if mutable else 'const ') + f'NlrOwned{tag[1]} *' if self.language == 'c' else
                    ('&mut ' if mutable else '&') + f'NlrOwned{tag[1]}')
        if isinstance(tag, tuple) and tag[0] == 'owner':
            return f'NlrOwned{tag[1]}'
        return super().type(tag)

    def inline_add_statement(self, target, value, indent):
        if id(getattr(self, 'function', None)) in getattr(self, 'scratch_functions', set()):
            return self.scratch_add_statement(target, value, indent)
        name = self.expression(target)
        a, b = name + '_a', name + '_b'
        low, high, boundary = name + '_low', name + '_high', name + '_boundary'
        self.line(f'let {a}: int = {self.expression(value.args[0])}', indent)
        self.line(f'let {b}: int = {self.expression(value.args[1])}', indent)
        self.line(f'let {low}: int = (- -9223372036854775807 1)', indent)
        self.line(f'let {high}: int = 9223372036854775807', indent)
        self.line(f'let mut {name}: int = 0', indent)
        # I use exactly the helper formulas; only result assignment replaces return.
        from nanoisa_reconstruction import NANO_INTEGER_HELPERS
        helper = NANO_INTEGER_HELPERS['add']
        positive = 'return (+ low (- (- a boundary) 1))'
        negative = 'return (+ high (+ (- a boundary) 1))'
        require(positive in helper and negative in helper and 'return (+ a b)' in helper,
                'require unchanged guarded ADD formulas')
        def formula(text):
            import re
            names = {'a': a, 'b': b, 'low': low, 'high': high, 'boundary': boundary}
            return re.sub(r'\b(a|b|low|high|boundary)\b', lambda m: names[m[0]], text[7:])
        self.line(f'if (> {b} 0) {{', indent)
        self.line(f'let {boundary}: int = (- {high} {b})', indent + 1)
        self.line(f'if (> {a} {boundary}) {{ set {name} {formula(positive)} }} else {{ set {name} (+ {a} {b}) }}', indent + 1)
        self.line(f'}} else {{ if (< {b} 0) {{', indent)
        self.line(f'let {boundary}: int = (- {low} {b})', indent + 1)
        self.line(f'if (< {a} {boundary}) {{ set {name} {formula(negative)} }} else {{ set {name} (+ {a} {b}) }}', indent + 1)
        self.line(f'}} else {{ set {name} (+ {a} {b}) }} }}', indent)

    def scratch_add_statement(self, target, value, indent):
        # This private bank is not used by any expression or lasting result.
        name = self.expression(target)
        a, b, low, high, boundary = (f'nlr_rs_{part}' for part in ('a', 'b', 'low', 'high', 'boundary'))
        self.line(f'set {a} {self.expression(value.args[0])}', indent)
        self.line(f'set {b} {self.expression(value.args[1])}', indent)
        self.line(f'let mut {name}: int = 0', indent)
        from nanoisa_reconstruction import NANO_INTEGER_HELPERS
        helper = NANO_INTEGER_HELPERS['add']
        positive = 'return (+ low (- (- a boundary) 1))'
        negative = 'return (+ high (+ (- a boundary) 1))'
        require(positive in helper and negative in helper and 'return (+ a b)' in helper,
                'require unchanged shared-bank guarded ADD formulas')
        def formula(text):
            import re
            names = {'a': a, 'b': b, 'low': low, 'high': high, 'boundary': boundary}
            return re.sub(r'\b(a|b|low|high|boundary)\b', lambda m: names[m[0]], text[7:])
        self.line(f'if (> {b} 0) {{', indent)
        self.line(f'set {boundary} (- {high} {b})', indent + 1)
        self.line(f'if (> {a} {boundary}) {{ set {name} {formula(positive)} }} else {{ set {name} (+ {a} {b}) }}', indent + 1)
        self.line(f'}} else {{ if (< {b} 0) {{', indent)
        self.line(f'set {boundary} (- {low} {b})', indent + 1)
        self.line(f'if (< {a} {boundary}) {{ set {name} {formula(negative)} }} else {{ set {name} (+ {a} {b}) }}', indent + 1)
        self.line(f'}} else {{ set {name} (+ {a} {b}) }} }}', indent)

    def inline_div_statement(self, target, value, indent):
        name = self.expression(target)
        a, b, low = name + '_a', name + '_b', name + '_low'
        self.line(f'let {a}: int = {self.expression(value.args[0])}', indent)
        self.line(f'let {b}: int = {self.expression(value.args[1])}', indent)
        self.line(f'let {low}: int = (- -9223372036854775807 1)', indent)
        self.line(f'let mut {name}: int = 0', indent)
        self.line(f'if (!= {b} 0) {{', indent)
        self.line(f'if (and (== {a} {low}) (== {b} -1)) {{', indent + 1)
        self.line(f'set {name} {low}', indent + 2)
        self.line('} else {', indent + 1)
        self.line(f'set {name} (/ {a} {b})', indent + 2)
        self.line('}', indent + 1)
        self.line('}', indent)

    def inline_sub_statement(self, target, value, indent):
        name = self.expression(target)
        a, b = name + '_a', name + '_b'
        low, high, boundary = name + '_low', name + '_high', name + '_boundary'
        self.line(f'let {a}: int = {self.expression(value.args[0])}', indent)
        self.line(f'let {b}: int = {self.expression(value.args[1])}', indent)
        self.line(f'let {low}: int = (- -9223372036854775807 1)', indent)
        self.line(f'let {high}: int = 9223372036854775807', indent)
        self.line(f'let mut {name}: int = 0', indent)
        from nanoisa_reconstruction import NANO_INTEGER_HELPERS
        helper = NANO_INTEGER_HELPERS['sub']
        positive = 'return (+ high (+ (- a boundary) 1))'
        negative = 'return (+ low (- a (+ boundary 1)))'
        require(positive in helper and negative in helper and 'return (- a b)' in helper,
                'require unchanged guarded SUB formulas')
        def formula(text):
            import re
            names = {'a': a, 'b': b, 'low': low, 'high': high, 'boundary': boundary}
            return re.sub(r'\b(a|b|low|high|boundary)\b', lambda m: names[m[0]], text[7:])
        self.line(f'if (> {b} 0) {{', indent)
        self.line(f'let {boundary}: int = (+ {low} {b})', indent + 1)
        self.line(f'if (< {a} {boundary}) {{ set {name} {formula(positive)} }} else {{ set {name} (- {a} {b}) }}', indent + 1)
        self.line(f'}} else {{ if (< {b} 0) {{', indent)
        self.line(f'let {boundary}: int = (+ {high} {b})', indent + 1)
        self.line(f'if (> {a} {boundary}) {{ set {name} {formula(negative)} }} else {{ set {name} (- {a} {b}) }}', indent + 1)
        self.line(f'}} else {{ set {name} (- {a} {b}) }} }}', indent)

    def inline_mul_statement(self, target, value, indent):
        from nanoisa_reconstruction import NANO_INTEGER_HELPERS
        require(NANO_INTEGER_HELPERS['mul'].split('\nshadow ', 1)[0] == '''fn nlr_i64_mul(a: int, b: int) -> int {
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
}''', 'require the unchanged total signed-digit MUL algorithm')
        def temp(suffix):
            return Expr(INT, 'temporary', f'{target.value}_mul_{suffix}')
        a, b, factor, remaining, digit = (temp(s) for s in ('a', 'b', 'factor', 'remaining', 'digit'))
        name = self.expression
        for operand, original in ((a, value.args[0]), (b, value.args[1])):
            self.line(f'let {name(operand)}: int = {name(original)}', indent)
        self.line(f'let mut {name(factor)}: int = {name(a)}', indent)
        self.line(f'let mut {name(remaining)}: int = {name(b)}', indent)
        self.line(f'let mut {name(target)}: int = 0', indent)
        self.line(f'while (!= {name(remaining)} 0) {{', indent)
        self.line(f'let {name(digit)}: int = (% {name(remaining)} 2)', indent + 1)
        for op, compare in (('add', '>'), ('sub', '<')):
            result = temp(op)
            self.line(f'if ({compare} {name(digit)} 0) {{', indent + 1)
            renderer = self.inline_add_statement if op == 'add' else self.inline_sub_statement
            renderer(result, Expr(INT, 'arithmetic', op, (target, factor)), indent + 2)
            self.line(f'set {name(target)} {name(result)}', indent + 2)
            self.line('}', indent + 1)
        self.line(f'set {name(remaining)} (/ {name(remaining)} 2)', indent + 1)
        doubled = temp('double')
        self.inline_add_statement(doubled, Expr(INT, 'arithmetic', 'add', (factor, factor)), indent + 1)
        self.line(f'set {name(factor)} {name(doubled)}', indent + 1)
        self.line('}', indent)

    def statements(self, statements, indent=1):
        c = self.language == 'c'
        if not c and statements is self.function.body and indent == 1 and id(self.function) in self.scratch_functions:
            self.line('let mut nlr_rs_a: int = 0', indent)
            self.line('let mut nlr_rs_b: int = 0', indent)
            self.line('let nlr_rs_low: int = (- -9223372036854775807 1)', indent)
            self.line('let nlr_rs_high: int = 9223372036854775807', indent)
            self.line('let mut nlr_rs_boundary: int = 0', indent)
        if c and (self.shared_paths or self.single_shared_path or self.exclusive_control) and statements is self.function.body and indent == 1:
            for parameter in range(len(self.function.params)):
                self.line(f'(void)nlr_a{parameter};', indent)
        if c and statements is self.function.body and indent == 1:
            for parameter, tag in enumerate(self.function.params):
                if tag == STRING:
                    self.line(f'(void)nlr_a{parameter};', indent)
        for stmt in statements:
            kind = stmt[0]
            if kind == 'print_string':
                value = self.expression(stmt[1])
                if c:
                    self.line(f'(void)fwrite(({value}).data, 1, ({value}).length, stdout);', indent)
                    if stmt[2]: self.line("(void)fputc('\\n', stdout);", indent)
                else:
                    self.line(f'({"println" if stmt[2] else "print"} {value})', indent)
            elif kind in ('break', 'continue'):
                require(self.loop_exits, 'require an exit-aware structured transfer')
                self.line(kind + (';' if c else ''), indent)
            elif kind == 'let' and self.inline_div and stmt[2].kind == 'arithmetic' and stmt[2].value == 'div':
                self.inline_div_statement(stmt[1], stmt[2], indent)
            elif kind == 'let' and self.inline_add and stmt[2].kind == 'arithmetic' and stmt[2].value == 'add':
                self.inline_add_statement(stmt[1], stmt[2], indent)
            elif kind == 'let' and self.inline_sub and stmt[2].kind == 'arithmetic' and stmt[2].value == 'sub':
                self.inline_sub_statement(stmt[1], stmt[2], indent)
            elif kind == 'let' and self.inline_mul and stmt[2].kind == 'arithmetic' and stmt[2].value == 'mul':
                self.inline_mul_statement(stmt[1], stmt[2], indent)
            elif kind in ('path_read', 'path_borrow_call'):
                target, root = stmt[1], self.local(stmt[2])
                if kind == 'path_read':
                    value = root + ''.join(f'.f{field}' for field in stmt[3]) + f'.f{stmt[4]}'
                else:
                    places = [root + ''.join(f'.f{field}' for field in path) for path in stmt[3]]
                    value = f'{self.name(1)}(' + ', '.join('&' + place for place in places) + ')' if c else f'({self.name(1)} ' + ' '.join(('&mut ' if self.functions[1].params[parameter][2] == 2 else '&') + place for parameter, place in enumerate(places)) + ')'
                self.line(f'{self.type(target.tag)} {self.expression(target)} = {value};' if c else
                          f'let {self.expression(target)}: {self.type(target.tag)} = {value}', indent)
                if c:
                    self.line(f'(void){self.expression(target)};', indent)
            elif kind == 'multi_borrow_write':
                _, slot, field, value = stmt
                self.line(f'nlr_a{slot}->f{field} = {self.expression(value)};' if c else
                          f'set nlr_a{slot}.f{field} {self.expression(value)}', indent)
            elif kind in ('multi_borrow_read', 'multi_borrow_call'):
                target = stmt[1]
                if kind == 'multi_borrow_read':
                    value = f'nlr_a{stmt[2]}->f{stmt[3]}' if c else f'nlr_a{stmt[2]}.f{stmt[3]}'
                else:
                    arguments = [('' if c else ('mut ' if parameter != 1 else '')) + self.local(slot) for parameter, slot in enumerate(stmt[2])]
                    value = f'{self.name(1)}(' + ', '.join('&' + arg for arg in arguments) + ')' if c else f'({self.name(1)} ' + ' '.join('&' + arg for arg in arguments) + ')'
                self.line(f'{self.type(target.tag)} {self.expression(target)} = {value};' if c else
                          f'let {self.expression(target)}: {self.type(target.tag)} = {value}', indent)
                if c:
                    self.line(f'(void){self.expression(target)};', indent)
            elif kind == 'borrow_write':
                _, field, value = stmt
                self.line(f'nlr_a0->f{field} = {self.expression(value)};' if c else
                          f'set nlr_a0.f{field} {self.expression(value)}', indent)
            elif kind == 'while_observe':
                _, condition, observations, loop_body = stmt
                self.line('while (true) {' if c else 'while true {', indent)
                self.statements(observations, indent + 1)
                value = self.expression(condition)
                self.line(f'if (!({value})) {{ break; }}' if c else
                          f'if (not {value}) {{ break }}', indent + 1)
                self.statements(loop_body, indent + 1)
                self.line('}', indent)
            elif kind in ('borrow_read', 'borrow_call'):
                _, target, operand = stmt
                if kind == 'borrow_read':
                    value = f'nlr_a0->f{operand}' if c else f'nlr_a0.f{operand}'
                else:
                    value = f'{self.name(1)}(&{self.local(operand)})' if c else f'({self.name(1)} {"&mut " if self.exclusive else "&"}{self.local(operand)})'
                self.line(f'{self.type(target.tag)} {self.expression(target)} = {value};' if c else
                          f'let {self.expression(target)}: {self.type(target.tag)} = {value}', indent)
                if c:
                    self.line(f'(void){self.expression(target)};', indent)
            elif kind == 'call_void':
                self.line(self.expression(stmt[1]) + (';' if c else ''), indent)
            elif kind == 'return_void':
                self.line('return;' if c else 'return', indent)
            elif kind == 'assert':
                expression = self.expression(stmt[1])
                flush = '(void)fflush(stdout); ' if c and getattr(self, 'has_strings', False) else ''
                self.line(f'if (!({expression})) {{ {flush}fputs("Assertion failed\\n", stderr); abort(); }}' if c else
                          f'assert {expression}', indent)
            elif kind == 'own_parameter':
                _, slot, layout = stmt
                name, typ = self.local(slot), self.type(owned_type(layout))
                self.line(f'{typ} {name} = nlr_a{slot};' if c else
                          f'let {name}: {typ} = nlr_a{slot}', indent)
            elif kind == 'own_pack':
                _, target, layout, values = stmt
                fields = ', '.join((f'.f{i} = ' if c else f'f{i}: ') + self.expression(v)
                                   for i, v in enumerate(values))
                name, typ = self.expression(target), self.type(target.tag)
                self.line(f'{typ} {name} = ({typ}){{{fields}}};' if c else
                          f'let {name}: {typ} = {typ} {{ {fields} }}', indent)
            elif kind == 'own_pack_store':
                require(not c, 'require Nano-only planned owner construction')
                _, slot, layout, values = stmt
                fields = ', '.join(f'f{i}: ' + self.expression(value) for i, value in enumerate(values))
                self.line(f'let mut {self.local(slot)}: NlrOwned{layout} = NlrOwned{layout} {{ {fields} }}', indent)
            elif kind == 'own_move_store':
                require(not c, 'require Nano-only planned owner transfer')
                _, slot, layout, source, assignment = stmt
                self.line(f'set {self.local(slot)} {self.local(source)}' if assignment else
                          f'let mut {self.local(slot)}: NlrOwned{layout} = {self.local(source)}', indent)
            elif kind == 'own_read_store':
                require(not c, 'require Nano-only planned scalar observation')
                _, slot, owner, field = stmt
                self.line(f'set {self.local(slot)} {self.local(owner)}.f{field}', indent)
            elif kind == 'own_assign':
                _, slot, _, value = stmt
                self.line(f'{self.local(slot)} = {self.expression(value)};' if c else
                          f'set {self.local(slot)} {self.expression(value)}', indent)
            elif kind in ('own_store', 'own_move'):
                if kind == 'own_store':
                    _, slot, layout, value = stmt
                    name, typ, value = self.local(slot), f'NlrOwned{layout}', self.expression(value)
                else:
                    _, target, slot = stmt
                    name, typ, value = self.expression(target), self.type(target.tag), self.local(slot)
                self.line(f'{typ} {name} = {value};' if c else f'let {"mut " if self.exclusive else ""}{name}: {typ} = {value}', indent)
            elif kind == 'own_read':
                _, target, slot, field = stmt
                name, typ, value = self.expression(target), self.type(target.tag), f'{self.local(slot)}.f{field}'
                self.line(f'{typ} {name} = {value};' if c else f'let {name}: {typ} = {value}', indent)
                if c:
                    self.line(f'(void){name};', indent)
            elif kind in ('own_unpack', 'own_unpack_pruned'):
                _, slot, layout, targets = stmt[:4]
                if kind == 'own_unpack_pruned':
                    require(not c and len(stmt[4]) == len(targets) and
                            all(type(keep) is bool and (keep or target.tag in (INT, BOOL))
                                for target, keep in zip(targets, stmt[4])),
                            'require only unused scalar unpack copies to be omitted')
                if not c:
                    fields = ', '.join(f'f{i}' for i in range(len(targets)))
                    self.line(f'let NlrOwned{layout} {{ {fields} }} = {self.local(slot)}', indent)
                for i, target in enumerate(targets):
                    if kind == 'own_unpack_pruned' and not stmt[4][i]:
                        continue
                    name, typ = self.expression(target), self.type(target.tag)
                    self.line(f'{typ} {name} = {self.local(slot)}.f{i};' if c else
                              f'let {name}: {typ} = f{i}', indent)
                    if c:
                        self.line(f'(void){name};', indent)
            else:
                super().statements([stmt], indent)

    def multi_shadow_local_cost(self):
        # I count both independent owner sets, call results and pattern captures.
        return 14 + 6 * sum(len(vector) for vector in self.witness)

    def multi_shadow(self):
        require(self.multi_shadow_local_cost() <= 128, 'limit generated multi-reference shadow to 128 local slots')
        body = f'\nshadow {self.name(1)} {{\n'
        for suffix in ('first', 'second'):
            for parameter, vector in enumerate(self.witness):
                layout = self.functions[1].params[parameter][1]
                fields = ', '.join(f'f{i}: {self.expression(value)}' for i, value in enumerate(vector))
                body += f'    let {"mut " if parameter != 1 else ""}owner_{suffix}_{parameter}: NlrOwned{layout} = NlrOwned{layout} {{ {fields} }}\n'
            arguments = ' '.join(f'{"&mut " if p != 1 else "&"}owner_{suffix}_{p}' for p in range(3))
            body += f'    let result_{suffix}: {self.type(self.functions[1].result)} = ({self.name(1)} {arguments})\n'
            for parameter, vector in enumerate(self.witness):
                layout = self.functions[1].params[parameter][1]
                fields = ', '.join(f'f{i}' for i in range(len(vector)))
                body += f'    let NlrOwned{layout} {{ {fields} }} = owner_{suffix}_{parameter}\n'
                for i, value in enumerate(vector):
                    body += f'    let field_{suffix}_{parameter}_{i}: {self.type(value.tag)} = f{i}\n'
        body += '    assert (== result_first result_second)\n'
        for parameter, vector in enumerate(self.witness):
            for i in range(len(vector)):
                body += f'    assert (== field_first_{parameter}_{i} field_second_{parameter}_{i})\n'
        return body + '}\n'

    def run(self, entry):
        c = self.language == 'c'
        string_literals = literals_in(self.functions)
        self.has_strings = bool(string_literals) or any(STRING in fn.params for fn in self.functions)
        if self.inline_add or self.has_strings and not c:
            require(all(self.nano_local_cost(fn) <= 128 for fn in self.functions),
                    'limit generated exclusive Nano functions to 128 local slots')
        declarations = ['#include <stdint.h>\n#include <stdbool.h>\n#include <stdio.h>\n#include <stdlib.h>'] if c else []
        if c and self.has_strings:
            declarations.append('typedef struct { const unsigned char *data; size_t length; } NlrString;')
            for index, data in string_literals.items():
                declarations.append(f'static const unsigned char nlr_string_{index}[] = {{' +
                                    ', '.join(str(byte) for byte in data or b'\0') + '};')
        for i, layout in enumerate(self.layouts):
            fields = (' '.join(f'{self.type(t)} f{j};' for j, t in enumerate(self.fields[i])) if c else
                      ', '.join(f'f{j}: {self.type(t)}' for j, t in enumerate(self.fields[i])))
            declarations.append(f'typedef struct {{ {fields} }} NlrOwned{i};' if c else
                                f'resource struct NlrOwned{i} {{ {fields} }}')
        body = super().run(entry).replace(
            '# I reconstruct executable scalar regions; original shadows are not retained.',
            '# I reconstruct ordered resource transfers; original shadows are not retained.')
        if not c:
            # I emit the retained entry directly; an extra wrapper exceeds the producer budget.
            wrapper = 'fn main() -> int {\n    return (main)\n}\n'
            require(body.endswith(wrapper), 'require the exact generated entry wrapper')
            body = body[:-len(wrapper)]
            require(sum(line.startswith('fn ') for line in body.splitlines()) <= 7,
                    'reserve one of eight producer functions for shadow execution')
        borrowed = (len(self.functions) == 2 and self.functions[1].params and
                    isinstance(self.functions[1].params[0], tuple) and
                    self.functions[1].params[0][0] == 'borrow')
        if not c and borrowed:
            require(sum(line.startswith('fn ') for line in body.splitlines()) == 2,
                    'require the existing two-function borrowed Nano profile')
            layout = self.functions[1].params[0][1]
            fields = self.fields[layout]
            values = ['0' if tag == INT else 'false' for tag in fields]
            entries = ', '.join(f'f{i}: {value}' for i, value in enumerate(values))
            names = ', '.join(f'f{i}' for i in range(len(fields)))
            if self.paths or self.exclusive_control:
                body += self.path_shadow()
            elif self.multi:
                body += self.multi_shadow()
            elif self.exclusive:
                body += f'\nshadow {self.name(1)} {{\n'
                for suffix in ('first', 'second'):
                    body += f'    let mut owner_{suffix}: NlrOwned{layout} = NlrOwned{layout} {{ {entries} }}\n'
                    body += f'    let result_{suffix}: {self.type(self.functions[1].result)} = ({self.name(1)} &mut owner_{suffix})\n'
                    body += f'    let NlrOwned{layout} {{ {names} }} = owner_{suffix}\n'
                    for i, tag in enumerate(fields):
                        body += f'    let field_{suffix}_{i}: {self.type(tag)} = f{i}\n'
                body += '    assert (== result_first result_second)\n'
                for i in range(len(fields)):
                    body += f'    assert (== field_first_{i} field_second_{i})\n'
                body += '}\n'
            else:
                body += f'\nshadow {self.name(1)} {{\n    let owner: NlrOwned{layout} = NlrOwned{layout} {{ {entries} }}\n'
                body += f'    let observed: {self.type(self.functions[1].result)} = ({self.name(1)} &owner)\n'
                body += f'    assert (== observed ({self.name(1)} &owner))\n'
                body += f'    let NlrOwned{layout} {{ {names} }} = owner\n'
                for i, value in enumerate(values):
                    body += f'    assert (== f{i} {value})\n'
                body += '}\n'
        return '\n'.join(declarations) + '\n' + body


def reconstruct_owned(module, language):
    require(sum(len(fn['code']) for fn in module['functions']) <= 4096,
            'limit the complete owned module to 4096 instructions')
    analyses = [OwnedAnalyze(module, i) for i in range(len(module['functions']))]
    functions = [analysis.run() for analysis in analyses]
    visiting, visited = set(), set()
    def visit(index):
        require(index not in visiting, 'require acyclic direct resource calls')
        if index in visited:
            return
        visiting.add(index)
        for target in analyses[index].calls:
            visit(target)
        visiting.remove(index)
        visited.add(index)
    for index in range(len(functions)):
        visit(index)
    return OwnedEmit(functions, language, module['layouts']).run(module['entry'])
