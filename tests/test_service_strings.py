"""I carry counted strings through service calls without host-pointer values."""
import os
from pathlib import Path
import shlex
import struct
import zlib
import unittest
from tests import test_service_drivers as support

ROOT=support.ROOT
SOURCE=r'''service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"
pure fn identity(text:string)->string{return text}
shadow identity {assert (== (identity "a\0b") "a\0b") assert (!= (identity "a\0b") "a\0c")}
fn apply(f:fn(string)->string,text:string)->string{return (f text)}
shadow apply {assert (== (apply identity "hello") "hello")}
fn choose(flag:bool,left:string,right:string)->string{if flag{return left}return right}
shadow choose {assert (== (choose false "a" "") "")}
fn main()->int {
 let saved:string=(apply identity "a\0b")
 let mut current:string=""
 let mut i:int=0
 while (< i 20) {set current (choose (== (% i 2) 0) saved "second") set i (+ i 1)}
 assert (== current "second")
 assert (== (identity "nsi:nanolang/filesystem#File") "nsi:nanolang/filesystem#File")
 assert (== (identity "main") "main") assert (== (identity "Ok") "Ok")
 assert (== (str_length saved) 3) assert (== (str_length "") 0) assert (== (str_length "é") 2)
 assert (== saved "a\0b") assert (!= saved "a") assert (!= saved "a\0c")
 assert (== (identity "line\nnext\t\\end") "line\nnext\t\\end")
 match (temp) {Error(e)=>{return -1} Ok(file)=>{
  let retained:string=(apply identity saved)
  match (close file) {Error(e)=>{return -2} Ok()=>{}}
  assert (== retained saved)
 }}
 return 0
}
'''

class ServiceStrings(unittest.TestCase):
    setUp=support.ServiceDrivers.setUp
    run_command=support.ServiceDrivers.run_command

    def selected_drivers(self):
        self.env["NANOLANG_ROOT"]=str(ROOT)
        drivers=list(self.drivers)
        if os.environ.get('NANO_STRING_DRIVER'):
            wrapper=self.work/'updated-nano'
            wrapper.write_text('#!/bin/sh\nexec '+shlex.join(shlex.split(os.environ['NANO_STRING_DRIVER']))+' "$@"\n')
            wrapper.chmod(0o700);drivers.append(wrapper)
        return drivers

    def exercise(self,text,flags):
        self.source.write_text(text)
        if len(flags)>1: mode=['--services','--service-instruction-limit','1000000']
        elif flags==['--allow-tcp-connections']: mode=['--socket-instruction-limit','1000000']
        else: mode=['--file-indirect','--file-instruction-limit','1000000']
        for driver in self.selected_drivers():
            with self.subTest(driver=str(driver)):
                wire=self.work/(driver.name+'.nvm')
                self.run_command([driver,self.source,*flags,'--emit-nvm','-o',wire])
                self.run_command([ROOT/'bin/nano_vm',*flags,*mode,wire])
                native=self.work/(driver.name+'.native')
                self.run_command([driver,self.source,*flags,'-o',native])
                self.run_command([native,*flags])
                self.run_command([native],1)

    def test_checked_source_vm_and_native(self):
        self.exercise(SOURCE,['--allow-temporary-files'])

    def test_tcp_catalog_strings(self):
        self.catalog.write_bytes((ROOT/'tests/fixtures/nsi_socket_plan.json').read_bytes())
        text=SOURCE[:SOURCE.index(' match (temp)')]+' return 0\n}\n'
        text=text.replace('nsi:nanolang/filesystem','nsi:nanolang/net').replace('#File','#Conn')
        self.exercise(text,['--allow-tcp-connections'])

    def test_mixed_catalog_strings(self):
        (self.work/'tcp.nsi.json').write_bytes((ROOT/'tests/fixtures/nsi_socket_plan.json').read_bytes())
        (self.work/'tcp.nano').write_text('service "nsi:nanolang/net" catalog 1 from "tcp.nsi.json"\npub fn text()->string{return "tcp"}\nshadow text {assert (== (text) "tcp")}\n')
        text='module "tcp.nano" as tcp\n'+SOURCE.replace(' let saved:string=', ' assert (== (tcp.text) "tcp")\n let saved:string=')
        self.exercise(text,['--allow-temporary-files','--allow-tcp-connections'])

    def test_wrong_string_types_preserve_output(self):
        flags=['--allow-temporary-files']
        for text in ['assert (== "a" 1)', 'let text:string=1', 'let size:int=(identity "a")', 'let n:int=(str_length 3)', 'let n:int=(str_length)']:
            self.source.write_text(SOURCE.replace('let saved:string=',text+'\n let saved:string='))
            for driver in self.selected_drivers():
                wire=self.work/'prior.nvm';wire.write_bytes(b'prior output')
                self.run_command([driver,self.source,*flags,'--emit-nvm','-o',wire],1)
                self.assertEqual(wire.read_bytes(),b'prior output')

    def test_invalid_literal_and_operand_types_preserve_output(self):
        self.source.write_text(support.DECL+'fn main()->int {assert (== "a" "a") return 0}\n')
        wire=self.work/'valid.nvm'
        self.run_command([ROOT/'bin/nano_virt',self.source,'--allow-temporary-files','--emit-nvm','-o',wire])
        original=wire.read_bytes()
        count=struct.unpack_from('<I',original,24)[0]
        sections=[struct.unpack_from('<IIQQ',original,40+24*i) for i in range(count)]
        code=next(offset for kind,flags,offset,size in sections if kind==6)
        self.assertEqual(original[code:code+11:5],bytes([4,4,68]))
        for label,offset,value in [('constant',code+1,b'\xff'*4),('type',code+10,b'\x20')]:
            changed=bytearray(original);changed[offset:offset+len(value)]=value
            struct.pack_into('<I',changed,36,zlib.crc32(changed[40:]))
            bad=self.work/(label+'.nvm');bad.write_bytes(changed)
            output=self.work/'prior.c';output.write_bytes(b'prior output')
            self.run_command([ROOT/'bin/nvm2c','--file-temporary','--file-indirect','--entry-name','check',bad,'-o',output],1)
            self.assertEqual(output.read_bytes(),b'prior output')
            self.run_command([ROOT/'bin/nano_vm','--allow-temporary-files','--file-indirect','--file-instruction-limit','1000000',bad],1)
