#!/usr/bin/env python3
"""I retain immutable C catalog views as pure NanoLang data, never runtime FFI."""
from pathlib import Path
import argparse,json,subprocess
ROOT=Path(__file__).resolve().parents[1]


def catalog_views(oracle):
    return subprocess.check_output([str(oracle)],cwd=ROOT,text=True).splitlines()


def parse_view(text):
    prefix,text=text.split(';',1);position=0
    def string():
        nonlocal position
        end=text.index(':',position);length=int(text[position:end]);position=end+1
        value=text[position:position+length];position+=length
        assert text[position]==';';position+=1;return value
    def number():
        nonlocal position
        assert text[position]=='#';end=text.index(';',position)
        value=int(text[position+1:end]);position=end+1;return value
    strings={(0,0,0,0):string()};types,methods=number(),number();numbers={}
    for i in range(types):
        for f in range(2):strings[1,i,f,0]=string()
        for f in range(2):numbers[1,i,f,0]=number()
        for j in range(numbers[1,i,1,0]):
            for f in range(2,5):strings[1,i,f,j]=string()
            numbers[1,i,2,j]=number()
    for i in range(methods):
        for f in range(4):strings[2,i,f,0]=string()
        for f in range(5):numbers[2,i,f,0]=number()
        for j in range(numbers[2,i,4,0]):
            for f in range(4,7):strings[2,i,f,j]=string()
            for f in range(5,10):numbers[2,i,f,j]=number()
        for j in range(2):numbers[2,i,10,j]=number();strings[2,i,7,j]=string()
    assert position==len(text)
    return types,methods,strings,numbers


def render(views):
    catalogs=[parse_view(v) for v in views];assert len(catalogs)==3
    lines=['module nvm2c_service_catalog','# I generate immutable data with scripts/generate_nvm2c_service_catalog.py.',
           '# I call no C catalog getter at runtime. Unknown selectors stay empty/-1.','',
           'pub fn count(catalog: int, kind: int) -> int {']
    for c,(types,methods,_,_) in enumerate(catalogs,1):
        lines += [f'    if (== catalog {c}) {{',f'        if (== kind 1) {{ return {types} }}',f'        if (== kind 2) {{ return {methods} }}','    }']
    lines+=['    return -1','}','shadow count { assert (== (count 1 1) 8) assert (== (count 3 2) 4) assert (== (count 0 1) -1) }','']
    def branch(rows,depth,indent):
        names=['kind','ordinal','field','member']
        if depth==4:
            assert len(rows)==1
            return [' '*indent+'return '+rows[0][1]]
        result=[]
        for key in sorted({k[depth] for k,v in rows}):
            result.append(' '*indent+f'if (== {names[depth]} {key}) {{')
            result+=branch([(k,v) for k,v in rows if k[depth]==key],depth+1,indent+4)
            result.append(' '*indent+'}')
        return result
    for name,index,result,fallback in [('text',2,'string','""'),('number',3,'int','-1')]:
        lines += [f'pub fn {name}(catalog: int, kind: int, ordinal: int, field: int, member: int) -> {result} {{']
        for c,data in enumerate(catalogs,1):
            rows=[(k,json.dumps(v) if name=='text' else str(v)) for k,v in data[index].items() if name!='text' or v]
            lines += [f'    if (== catalog {c}) {{',*branch(rows,0,8),'    }']
        lines += [f'    return {fallback}','}']
        if name=='text':lines += ['shadow text { assert (== (text 1 0 0 0 0) "nsi:nanolang/filesystem") assert (== (text 3 1 0 1 0) "Connection") assert (== (text -1 1 0 0 0) "") }','']
        else:lines += ['shadow number { assert (== (number 2 1 8 1 0) 7) assert (== (number 3 2 1 4 0) 4) assert (== (number 4 1 0 0 0) -1) }','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');parser.add_argument('--oracle',type=Path,default=ROOT/'obj/nvm2c_service_catalog_reference');args=parser.parse_args()
    data=render(catalog_views(args.oracle));path=ROOT/'src_nano/nvm2c/service_catalog.nano'
    if args.check:
        if not path.exists() or path.read_text()!=data:raise SystemExit('I require current pure NanoLang service catalog data; run this generator without --check.')
    else:path.write_text(data)


if __name__=='__main__':main()
