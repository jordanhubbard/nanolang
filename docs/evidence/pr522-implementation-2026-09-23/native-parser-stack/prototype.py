import re
from pathlib import Path
root=Path('/tmp/pr522-native-parser-stack')
text=(root/'compiler.c').read_text()
pattern=re.compile(r'^static (.+?) (nl_\w+)\(([^)\n]*)\)(;| \{)$',re.M)
functions={}
for m in pattern.finditer(text):
    params=m[3].split(', ') if m[3]!='void' else []
    functions[m[2]]=(m[1]=='nrec_t',[i for i,p in enumerate(params) if p.startswith('nrec_t ')])
lines=[];current=None;count=0
for line in text.splitlines():
    m=pattern.fullmatch(line)
    if m:
        ret,indices=functions[m[2]]
        params=m[3].split(', ') if m[3]!='void' else []
        for i in indices:params[i]=params[i].replace('nrec_t ', 'const nrec_t *',1)
        if ret:params.append('nrec_t *nout')
        line='static '+('void' if ret else m[1])+' '+m[2]+'('+(', '.join(params) if params else 'void')+')'+m[4]
        current=m[2] if m[4]==' {' else None
    else:
        if current:
            ret,indices=functions[current]
            for i in indices:line=re.sub(r'\ba'+str(i)+r'\b','(*a'+str(i)+')',line)
            if ret:line=line.replace('return nresult;', '*nout = nresult; return;')
        call=re.search(r'\b(nl_\w+)\(([^()]*)\)',line)
        if call and call[1] in functions:
            ret,indices=functions[call[1]]
            args=call[2].split(', ') if call[2] else []
            for i in indices:
                assert re.fullmatch(r'(?:r|rl)\[\d+\]|nresult',args[i]),(call[1],args[i])
                args[i]='&'+args[i]
            start=call.start()
            if ret:
                dest=re.search(r'(r\[\d+\]|nresult) = $',line[:start]);assert dest,line
                args.append('&'+dest[1]);start=dest.start()
            line=line[:start]+call[1]+'('+', '.join(args)+')'+line[call.end():]
            count+=1
        if line=='}':current=None
    lines.append(line)
(root/'compiler-pointer.c').write_text('\n'.join(lines)+'\n')
print('I transformed',count,'private prototype call sites.')
