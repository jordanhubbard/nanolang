"""I qualify ordered, portable and owned public C concatenation."""
import os
from pathlib import Path
import re
import unittest
from tests import test_public_c_nonfinite_format as formatting
ROOT=Path(__file__).resolve().parents[1]
class PublicCConcat(unittest.TestCase):
    setUp=formatting.PublicCFormat.setUp
    run_cmd=formatting.PublicCFormat.run_cmd
    emit=formatting.PublicCFormat.emit
    compile=formatting.PublicCFormat.compile

    def test_ordered_nested_operands_globals_loops_and_aliases(self):
        output,code=self.emit(ORDER)
        self.assertNotIn('static const char *nano_cb_0_string_concat(',code)
        self.assertNotIn('static const char* nano_strcat(',code)
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization)
                self.compile(output,exe,standard,optimization)
                self.assertEqual(self.run_cmd([exe]).stdout,'12\n3456\n78\n-10|1.25\n89\n')

    def test_exact_empty_long_bytes_across_ordinary_routes(self):
        long='abλ'*700
        source=BYTES.replace('LONG',long)
        output,_=self.emit(source)
        expected='|ab|cd|left-right|headtail|'+long+'END\n'
        for standard in ('c99','c11'):
            for optimization in ('-O0','-O2'):
                exe=self.work/(standard+optimization)
                self.compile(output,exe,standard,optimization)
                self.assertEqual(self.run_cmd([exe]).stdout,expected)
        sourcepath=self.work/'ordinary.nano'
        self.assertEqual(self.run_cmd([ROOT/'bin/nano',sourcepath]).stdout,expected)
        module=self.work/'ordinary.nvm'
        self.run_cmd([ROOT/'bin/nano_virt',sourcepath,'--emit-nvm','-o',module])
        self.run_cmd([ROOT/'bin/nano_vm','--verify-only',module])
        self.assertEqual(self.run_cmd([ROOT/'bin/nano_vm',module]).stdout,expected)

    def test_generated_ownership_and_checked_failures(self):
        output,code=self.emit('fn main()->int{return 0} shadow main{assert (== (main) 0)}')
        prefix=re.search(r'static const char \*(nano_cb_\d+_)string_concat',code).group(1)
        harness=self.work/'ownership.c';harness.write_text(FAILURES.replace('GENERATED',str(output)).replace('PREFIX',prefix))
        exe=self.work/'ownership';self.compile(harness,exe)
        self.run_cmd([exe,'normal'])
        for mode,diagnostic in [('allocation','I could not allocate my C scalar result.'),
            ('registration','I could not register my C scalar cleanup.'),
            ('sum','I exceeded my C concatenation length.'),
            ('size','I exceeded my C string allocation size.'),
            ('null','I require nonnull C concatenation operands.')]:
            result=self.run_cmd([exe,mode],expected=1);self.assertIn(diagnostic,result.stderr)

    def test_binding_exact_types_refusal_and_recovery(self):
        exe=self.work/'api'
        self.run_cmd([os.environ.get('CC','cc'),'-std=c99','-D_POSIX_C_SOURCE=200809L',
            '-Wall','-Wextra','-Werror','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',
            '-I',ROOT/'src',ROOT/'tests/test_public_c_concat_api.c','-o',exe])
        self.run_cmd([exe,self.work/'previous.c'])

ORDER='''let mut sequence:int=0
fn step(n:int)->string{set sequence (+ (* sequence 10) n) return (int_to_string n)}
shadow step{assert (== (step 1) "1")}
fn nano_strcat(value:int)->int{return value}
shadow nano_strcat{assert (== (nano_strcat 4) 4)}
fn nano_cb_0_string_concat(value:int)->int{return value}
shadow nano_cb_0_string_concat{assert (== (nano_cb_0_string_concat 5) 5)}
fn join(left:string,right:string)->string{return (str_concat left right)}
shadow join{assert (== (join "a" "b") "ab")}
let global:string=(+ (step 1) (step 2))
fn condition()->string{set sequence (+ sequence 1) return (int_to_string sequence)}
shadow condition{assert (!= (condition) "")}
fn main()->int{
 assert (== sequence 12) assert (== global "12")
 assert (== (nano_strcat 4) 4) assert (== (nano_cb_0_string_concat 5) 5)
 set sequence 0
 let nested:string=(str_concat (+ (step 3) (step 4)) (str_concat (step 5) (step 6)))
 assert (== sequence 3456) assert (== nested "3456")
 set sequence 0
 assert (!= (+ (step 1) (step 2)) (str_concat (step 3) (step 4)))
 assert (== sequence 1234)
 let retained:string=(join "7" "8")
 let alias:string=retained
 let mixed:string=(+ (int_to_string -10) (+ "|" (float_to_string 1.25)))
 let mut i:int=0
 while (< i 300){let temporary:string=(str_concat retained (int_to_string i)) assert (!= temporary "") set i (+ i 1)}
 set sequence 0
 while (!= (+ (condition) "") "3"){}
 assert (== sequence 3) assert (== retained "78") assert (== alias "78")
 (println global) (println nested) (println alias) (println mixed)
 (println (join "8" "9"))
 return 0
}
shadow main{assert true}
'''
BYTES=r'''fn main()->int{
 (print (str_concat "" "")) (print "|")
 (print (+ "" "ab")) (print "|")
 (print (str_concat "cd" "")) (print "|")
 (print (+ "left" "-right")) (print "|")
 (print (str_concat "head\0ignored" "tail")) (print "|")
 (println (str_concat "LONG" "END"))
 return 0
}
shadow main{assert true}
'''
FAILURES=r'''#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static int allocations,releases,registrations,fail_allocate,fail_register,mode;
static void *fixture_malloc(size_t n){++allocations;return fail_allocate?NULL:malloc(n);}
static void fixture_free(void *p){if(p)++releases;free(p);}
static int fixture_atexit(void(*f)(void)){++registrations;return fail_register?-1:atexit(f);}
#define malloc fixture_malloc
#define free fixture_free
#define atexit fixture_atexit
#define main generated_main
#include "GENERATED"
#undef main
#undef atexit
#undef free
#undef malloc
static void verify_exit(void){
 if(mode==1)assert(allocations==3&&releases==2&&registrations==1);
 else if(mode==2){assert(allocations==1&&releases==1&&registrations==1);assert(PREFIXfloat_text_registered==0);}
 else assert(allocations==0&&releases==0&&registrations==0);
 assert(PREFIXfloat_text_head==NULL);
}
int main(int argc,char **argv){
 assert(argc==2);
 if(strcmp(argv[1],"allocation")==0){mode=1;assert(atexit(verify_exit)==0);PREFIXint_text_new(4);PREFIXfloat_text_new(1.25);fail_allocate=1;PREFIXstring_concat("a","b");return 9;}
 if(strcmp(argv[1],"registration")==0){mode=2;assert(atexit(verify_exit)==0);fail_register=1;PREFIXstring_concat("a","b");return 9;}
 if(strcmp(argv[1],"sum")==0){assert(atexit(verify_exit)==0);PREFIXstring_length_sum(SIZE_MAX,1);return 9;}
 if(strcmp(argv[1],"size")==0){assert(atexit(verify_exit)==0);PREFIXtext_allocate(SIZE_MAX);return 9;}
 if(strcmp(argv[1],"null")==0){assert(atexit(verify_exit)==0);PREFIXstring_concat(NULL,"a");return 9;}
 assert(PREFIXstring_length_sum(SIZE_MAX-1,1)==SIZE_MAX);
 char left[]="left",right[]="right";
 const char *a=PREFIXstring_concat(left,right);const char *b=PREFIXstring_concat("","");
 const char *c=PREFIXint_text_new(-10);const char *d=PREFIXfloat_text_new(1.25);
 const char *e=PREFIXstring_concat(a,c);
 left[0]='L';right[0]='R';assert(strcmp(a,"leftright")==0&&strcmp(b,"")==0);
 assert(strcmp(c,"-10")==0&&strcmp(d,"1.25")==0&&strcmp(e,"leftright-10")==0);
 assert(allocations==5&&registrations==1);PREFIXfloat_text_cleanup();assert(releases==5);
 PREFIXfloat_text_cleanup();assert(releases==5&&PREFIXfloat_text_head==NULL);return 0;
}
'''
if __name__=='__main__':unittest.main()
