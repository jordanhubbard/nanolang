int scope_probe(void) { int value = 7; { __auto_type value = ({ int converted = value; converted; }); return value; } }
