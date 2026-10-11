from pathlib import Path
import subprocess,shlex
root=Path('/Users/jordanh/Src/nanolang');work=Path('/private/tmp/nl51-lookup-profile');(work/'bin').mkdir(parents=True,exist_ok=True)
s=(root/'src/env.c').read_text().replace('Function *env_get_function(Environment *env, const char *name) {','static Function *profile_lookup(Environment *env, const char *name) {',1)
s+='''\n#include <time.h>
static unsigned long long lookup_calls, lookup_ns;
void nl_lookup_profile_reset(void) { lookup_calls=lookup_ns=0; }
void nl_lookup_profile_report(const char *name) { fprintf(stderr,"lookup-profile %s %llu %llu\\n",name,lookup_calls,lookup_ns); }
Function *env_get_function(Environment *env, const char *name) {
 struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
 Function *r=profile_lookup(env,name); clock_gettime(CLOCK_MONOTONIC,&b);
 lookup_calls++;lookup_ns+=(b.tv_sec-a.tv_sec)*1000000000LL+b.tv_nsec-a.tv_nsec;return r;
}
''';(work/'env.c').write_text(s)
s=(root/'src/eval.c').read_text();s='extern void nl_lookup_profile_reset(void);\nextern void nl_lookup_profile_report(const char *);\n'+s
s=s.replace('eval_statement(item->as.shadow.body, env);','nl_lookup_profile_reset();\n                eval_statement(item->as.shadow.body, env);\n                nl_lookup_profile_report(func_name);',1);(work/'eval.c').write_text(s)
raw=subprocess.check_output(['make','-f','Makefile.gnu','-Bn','bin/nanoc_c'],cwd=root,text=True)
replacements={};link=None
for line in raw.replace('\\\n',' ').splitlines():
 try:args=shlex.split(line)
 except ValueError:continue
 if '-c' in args and '-o' in args and args[args.index('-c')+1] in {'src/env.c','src/eval.c'}:
  source=args[args.index('-c')+1];out=args[args.index('-o')+1];target=work/(Path(source).stem+'.o');args[args.index('-c')+1]=str(work/Path(source).name);args[args.index('-o')+1]=str(target);subprocess.run(args+['-Isrc'],cwd=root,check=True);replacements[out]=str(target)
 lex=shlex.shlex(line,posix=True,punctuation_chars=';');lex.whitespace_split=True
 try:tokens=list(lex)
 except ValueError:continue
 for i,t in enumerate(tokens):
  if t!='cc':continue
  end=next((j for j in range(i,len(tokens)) if tokens[j]==';'),len(tokens));candidate=tokens[i:end]
  if '-o' in candidate and candidate[candidate.index('-o')+1]=='bin/nanoc_c':link=candidate
assert len(replacements)==2 and link
link=[replacements.get(a,a) for a in link];link[link.index('-o')+1]=str(work/'bin/nanoc_c');subprocess.run(link,cwd=root,check=True)
for name in ['src','lib','modules','stdlib','include']:
 if not (work/name).exists():(work/name).symlink_to(root/name)
