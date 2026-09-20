/* I distinguish real host calls from explicitly injected trusted-hook failures. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [libraryPath, wasmPath, vectorsPath] = process.argv.slice(2);
const host = await import(pathToFileURL(libraryPath));
const wasm = fs.readFileSync(wasmPath), spec = JSON.parse(fs.readFileSync(vectorsPath));
let checks = 0, realVectors = 0, modeled = 0;
function eq(a,b) { ++checks; assert.deepEqual(a,b); }
function bad(fn,kind=Error) { ++checks; assert.throws(fn,kind); }
function put(api,bytes) { for(let i=0;i<bytes.length;i++)api.call('put',i,bytes[i]); }
function start(api,path,fill=1) { put(api,path);api.call('init',path.length,fill); }
function done(api) {eq(api.call('roots_ok'),1);eq(api.call('finish'),1);eq(api.close(),0);eq(api.close(),0);bad(()=>api.call('pages'));}
function make(paths) {
    const Actual=WebAssembly.Instance;let instance;
    WebAssembly.Instance=class extends Actual {constructor(...args){super(...args);instance=this;}};
    try {return {api:host.createReadTextInstance(wasm,paths),instance};}
    finally {WebAssembly.Instance=Actual;}
}
function checksum(bytes) {let h=2166136261;for(const b of bytes)h=Math.imul(h^b,16777619)>>>0;return h;}
const small=spec.vectors.find(x=>x.name==='small'), big=spec.vectors.find(x=>x.name==='exact');
for(const v of spec.vectors) {
    const path=Buffer.from(v.pathHex,'hex'), {api}=make([path]);start(api,path);
    const before=api.call('pages');eq(api.call('read',0,0),v.status<<8);++realVectors;
    if(!v.status){eq(api.call('length',0),v.length);eq(api.call('hash',0)>>>0,v.hash);}
    if(v.name==='exact'){++checks;assert(api.call('pages')>before);}
    eq(api.report().closeAttempted,v.name!=='missing');done(api);
}
// I copy the allowlist, preserve an earlier result across scratch reuse, and isolate instances.
{
    const original=Buffer.from(small.pathHex,'hex'), copy=Buffer.from(original), rows=Array.from({length:64},(_,i)=>i===63?copy:Buffer.from('deny'+i));
    const {api}=make(rows);copy.fill(33);rows.fill(Buffer.from('changed'));
    start(api,original);eq(api.call('read',0,0),0);
    const second=Buffer.from('second');fs.writeFileSync(wasmPath+'.copy-second.bin',second);
    try {fs.writeFileSync(original,second);eq(api.call('read',1,0),0);}
    finally {fs.writeFileSync(original,Buffer.from('copied'));}
    eq(api.call('hash',0)>>>0,small.hash);eq(api.call('hash',1)>>>0,checksum(second));
    const other=make([]).api;start(other,original);eq(other.call('read',0,0),256);done(other);done(api);
    bad(()=>make(Array.from({length:65},()=>original)));bad(()=>make([Buffer.from([97,0,98])]));bad(()=>make([Buffer.alloc(0)]));
}
for(const mode of [1,2,3,4,5]) {
    const {api}=make([Buffer.from(small.pathHex,'hex')]);start(api,Buffer.from(small.pathHex,'hex'));
    eq(api.call('read',0,mode),[0,6,6,5,6,1][mode]);eq(api.report().opened,false);done(api);
}
for(const size of [0,4096,4097]) {
    const path=Buffer.alloc(size,113), {api}=make(size===4096?[path]:[]);start(api,path);
    let opens=0,seen=0;const original=fs.openSync;
    fs.openSync=(p,...args)=>{opens++;seen=p.length;return original(p,...args);};
    try {eq(api.call('read',0,0),size===4097?512:size===0?256:0);}
    finally {fs.openSync=original;}
    eq(opens,size===4096?1:0);if(size===4096)eq(seen,4096);done(api);
}
{
    const path=Buffer.concat([Buffer.from(small.pathHex,'hex'),Buffer.from([0,120,121])]),{api}=make([Buffer.from(small.pathHex,'hex')]);
    start(api,path);eq(api.call('read',0,0),0);eq(api.call('hash',0)>>>0,small.hash);done(api);
}
// I force all three managed publication allocation prefixes, then recover freshly.
{
    const path=Buffer.from(small.pathHex,'hex'), probe=make([path]).api;start(probe,path,8);
    if(probe.call('testing')) {
        probe.call('budget',100);eq(probe.call('read',0,0),0);const measured=100-probe.call('remaining');eq(measured,3);probe.call('budget',0xffffffff);done(probe);
        for(let i=0;i<measured;i++) {const api=make([path]).api;start(api,path,8);api.call('budget',i);eq(api.call('read',0,0),3);eq(api.call('roots_ok'),1);api.call('budget',0xffffffff);eq(api.call('read',0,0),0);done(api);++modeled;}
    } else done(probe);
}
// I observe real memory growth between calls and actual max-memory refusal.
for(const exhaust of [false,true]) {
    const path=Buffer.from((exhaust?big:small).pathHex,'hex'),api=make([path]).api;start(api,path);
    const pages=api.call('pages');eq(api.call('grow',exhaust?1024-pages:1),pages);
    eq(api.call('read',0,0),exhaust?3:0);done(api);
}
function rawSetup(api,path) {
    const base=api.call('raw_base')>>>0;for(let i=0;i<path.length;i++)api.call('raw_put',i,path[i]);
    const out=base+7001;for(let i=0;i<4;i++)api.call('raw_put',7001+i,i?0:91);
    return {base,out,args:[base,path.length,base+5000,16,out]};
}
function rawLength(api){let x=0;for(let i=0;i<4;i++)x+=api.call('raw_get',7001+i)*2**(8*i);return x;}
{
    const path=Buffer.from(small.pathHex,'hex'),{api}=make([path]),{base,out,args}=rawSetup(api,path),size=api.call('pages')*65536;
    const refusals=[[0xffffffff,2,args[2],16,out],[size,1,args[2],16,out],[base,path.length,base,16,out],
        [out,4,args[2],16,out],[base,path.length,out,4,out],[base,path.length,args[2],1048577,out],
        [base,4097,args[2],16,out],[base,path.length,args[2],16,size-3],[base,path.length,0xffffffff,16,out],[base,path.length,args[2],16,0xffffffff]];
    let opens=0;const open=fs.openSync;fs.openSync=(...a)=>{opens++;return open(...a);};
    try {for(const a of refusals){eq(api.call('raw_call',...a),4);eq(rawLength(api),91);}eq(opens,0);
        eq(api.call('raw_call',...args),0);eq(rawLength(api),small.length);eq(opens,1);}
    finally {fs.openSync=open;}
    eq(api.close(),0);
}
for(const v of [small,spec.vectors.find(x=>x.name==='empty')]) {
    const path=Buffer.from(v.pathHex,'hex'),api=make([path]).api,{args}=rawSetup(api,path);
    args[2]=api.call('pages')*65536;args[3]=0;
    eq(api.call('raw_call',...args),v.length?2:0);eq(rawLength(api),v.length?91:0);eq(api.close(),0);
}
// Real calls plus narrowly identified host faults; actual fd is closed once.
for(const mode of ['progress','close','limit-close','memory-read','memory-close','grow','active','publication','alloc0','alloc1','alloc2']) {
    const path=Buffer.from(small.pathHex,'hex'),{api,instance}=make([path]);const raw=rawSetup(api,path);
    const saved={open:fs.openSync,read:fs.readSync,close:fs.closeSync,from:Buffer.from,alloc:Buffer.alloc,U8:globalThis.Uint8Array};
    let opened=0,closed=0,fd=-1,reads=0,allocs=0,activeChecks=0;
    function ioerror(){return Object.assign(new Error('I inject a post-progress I/O error'),{code:'EIO',errno:-5});}
    function allocFail(){if(mode.startsWith('alloc') && allocs++===Number(mode.at(-1)))throw new RangeError('I inject allocation refusal');}
    fs.openSync=(...a)=>{opened++;fd=saved.open(...a);if(mode==='grow')instance.exports.memory.grow(1);
        if(mode==='active'){eq(api.close(),4);bad(()=>api.call('grow',1));activeChecks++;}return fd;};
    fs.readSync=(f,b,o,n,pos)=>{reads++;if(mode==='memory-read')throw new RangeError('I inject read allocation refusal');
        if(mode==='progress' && reads>1)throw ioerror();return saved.read(f,b,o,mode==='progress'?1:n,pos);};
    fs.closeSync=f=>{closed++;saved.close(f);if(mode==='memory-close')throw new RangeError('I inject close allocation refusal');if(mode==='close'||mode==='limit-close')throw ioerror();};
    Buffer.from=(...a)=>{allocFail();return saved.from(...a);};Buffer.alloc=(...a)=>{allocFail();return saved.alloc(...a);};
    if(mode==='publication')globalThis.Uint8Array=class extends saved.U8 {constructor(...a){if(a[0]===instance.exports.memory.buffer&&a[1]===raw.out)throw new RangeError('I inject length-view allocation refusal');super(...a);}};
    let result;
    try {const args=[...raw.args];if(mode==='limit-close')args[3]=1;result=api.call('raw_call',...args);}
    finally {fs.openSync=saved.open;fs.readSync=saved.read;fs.closeSync=saved.close;Buffer.from=saved.from;Buffer.alloc=saved.alloc;globalThis.Uint8Array=saved.U8;}
    const expected=mode==='limit-close'?2:mode==='grow'?4:(mode.startsWith('memory')||mode==='publication'||mode.startsWith('alloc'))?3:0;
    eq(result,expected);eq(rawLength(api),expected?91:mode==='progress'||mode==='close'?0:small.length);
    eq(opened,mode.startsWith('alloc')?0:1);eq(closed,opened);if(fd>=0)bad(()=>fs.fstatSync(fd));
    if(mode==='active')eq(activeChecks,1);
    if(mode==='progress')eq(api.report().bytesRead,1);
    if(mode==='limit-close')eq(api.report().closeError,true);
    eq(api.call('raw_call',...raw.args),0);eq(rawLength(api),small.length);eq(api.close(),0);++modeled;
}
// I validate exact envelopes before instantiation and leave body validity to the actual engine.
function leb(n){const b=[];do{let v=n&127;n=Math.floor(n/128);b.push(v|(n?128:0));}while(n);return b;}
function name(s){const b=[...Buffer.from(s)];return [...leb(b.length),...b];}
function section(id,b){return [id,...leb(b.length),...b];}
const type=section(1,[1,0x60,5,127,127,127,127,127,1,127]);
const imp=section(2,[1,...name('nanolang_host_v1'),...name('read_text'),0,0]);
const mem=section(5,[1,1,32,...leb(1024)]),exp=section(7,[1,...name('memory'),2,0]);
function moduleBytes(parts){return Buffer.from([0,97,115,109,1,0,0,0,...parts.flat()]);}
const minimal=moduleBytes([type,imp,mem,exp]);eq(host.createReadTextInstance(minimal,[]).close(),0);
const malformed=[moduleBytes([type,type,imp,mem,exp]),moduleBytes([imp,type,mem,exp]),
    moduleBytes([type,imp,mem,exp,section(8,[0])]),moduleBytes([type,imp,section(5,[1,3,32,...leb(1024)]),exp]),
    moduleBytes([section(1,[1,0x60,0,1,127]),imp,mem,exp]),moduleBytes([type,section(2,[2,...imp.slice(3)]),mem,exp]),
    moduleBytes([type,imp,section(5,[1,1,31,...leb(1024)]),exp]),moduleBytes([type,imp,mem]),
    moduleBytes([type,imp,section(3,[1,0]),mem,exp,section(10,[1,2,0,255])]),Buffer.concat([minimal,Buffer.from([0,128])])];
malformed.push(moduleBytes([type,section(2,[1,...name('wrong'),...name('read_text'),0,0]),mem,exp]),
    moduleBytes([type,imp,section(5,[1,5,32,...leb(1024)]),exp]),
    moduleBytes([section(1,[...leb(4097)]),imp,mem,exp]),
    moduleBytes([type,imp,section(3,[...leb(65536)]),mem,exp]),
    moduleBytes([type,imp,mem,exp,...Array.from({length:61},()=>section(0,[0]))]),
    Buffer.concat([minimal,Buffer.from([0,255,255,255,255,16])]));
for(let i=0;i<malformed.length;i++)fs.writeFileSync(wasmPath+'.node-negative-'+i+'.wasm',malformed[i]);
let opens=0;const open=fs.openSync;fs.openSync=(...a)=>{opens++;return open(...a);};
try {for(let i=0;i<malformed.length;i++)bad(()=>host.createReadTextInstance(malformed[i],[]),i===1||i===8?WebAssembly.CompileError:Error);eq(opens,0);}finally {fs.openSync=open;}
// I use a distinct raw trusted import only to probe defensive guest validation/latch paths.
for(const mode of ['status','unset','oversize','nul','reenter']) {
    let instance,calls=0;const path=Buffer.from(small.pathHex,'hex');
    instance=new WebAssembly.Instance(new WebAssembly.Module(wasm),{nanolang_host_v1:{read_text:(p,n,d,cap,out)=>{
        calls++;const view=new DataView(instance.exports.memory.buffer);
        if(mode==='reenter'){eq(instance.exports.read(1,0),1024);view.setUint32(out,0,true);return 0;}
        if(mode==='status')return 99;if(mode==='unset')return 0;
        view.setUint32(out,mode==='oversize'?cap+1:1,true);if(mode==='nul')new Uint8Array(instance.exports.memory.buffer)[d]=0;return 0;
    }}});
    const raw={call:(n,...a)=>instance.exports[n](...a)};start(raw,path);eq(raw.call('read',0,0),mode==='reenter'?0:1024);eq(calls,1);eq(raw.call('roots_ok'),1);eq(raw.call('finish'),1);++modeled;
}
{const api=make([]).api;start(api,Buffer.from(small.pathHex,'hex'));bad(()=>api.call('trap'));eq(api.report().terminal,true);bad(()=>api.call('pages'));eq(api.close(),0);}
console.log(JSON.stringify({engine:'Node',checks,realVectors,modeled,scope:'private direct Wasm adapter; no NanoISA admission'}));
