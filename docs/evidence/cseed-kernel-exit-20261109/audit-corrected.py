"""I replay the complete retained experiment; I do not launch a new factory."""
import argparse
import ast
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts import observe_cseed_kernel_exit as kernel
from scripts.compare_cseed_source_creation import shadow_phases,shadow_trace
PIN='ed64b3a8563305bd710689458b7dd9f9d243370e'
BASE='35a32bf746e9c7741158bb8e62563d92bde614f8'
sha=lambda data:hashlib.sha256(data).hexdigest()
@lru_cache(maxsize=None)
def git_bytes(pin,path):return subprocess.check_output(['git','show',pin+':'+path],cwd=ROOT)

def audit(archive,output,retained):
    assert not output.exists() and not retained.exists()
    old=ast.parse(git_bytes(PIN,'scripts/observe_cseed_kernel_exit.py'))
    new=ast.parse((ROOT/'scripts/observe_cseed_kernel_exit.py').read_text())
    for name in ('render_program','check_control','exit_intervals'):
        find=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(find(old))==ast.dump(find(new)),name
    inventory=[];texts={};images=[];objects=[];saved={}
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))
        for info in z.infolist():
            if info.is_dir():continue
            assert not info.filename.startswith('/') and '..' not in Path(info.filename).parts
            data=z.read(info)
            inventory.append({'path':info.filename,'bytes':len(data),'sha256':sha(data)})
            if data[:4]==b'\x7fELF':
                assert data[5]==1
                kind=struct.unpack_from('<H',data,16)[0];assert kind in (1,2,3)
                (objects if kind==1 else images).append({'path':info.filename,'bytes':len(data),'sha256':sha(data),'elf_type':kind})
                saved[info.filename]=data
            elif info.filename.endswith('/factory/kernel.jsonl'):
                pass  # I retain these exact bytes once in corrected-linux/kernel.jsonl.gz.
            else:texts[info.filename]=data.decode()
        prefix='.test_output/cseed-kernel-exit/'
        load=lambda name:json.loads(z.read(prefix+name))
        plan=load('plan.json');assert plan['source_pin']==PIN and plan['shadow_deadline_seconds']==10 and plan['factory_native_budget_seconds']==600
        shared=load('sources-before.json');assert len(shared)==197
        for path,expected in shared.items():assert sha(git_bytes(PIN,path))==sha(git_bytes(BASE,path))==expected
        observer_sources=load('observer-sources-before.json');assert len(observer_sources)==4
        for path,expected in observer_sources.items():assert sha(git_bytes(PIN,path))==expected
        providers=load('providers-before.json');assert len(providers)==6
        for path,expected in providers.items():assert sha(z.read(path))==expected
        control_life=load('control/lifecycle.json');factory_life=load('factory/lifecycle.json')
        for which,life in (('control',control_life),('factory',factory_life)):
            assert life['target_returncode']==life['target_final_returncode']==life['observer_returncode']==0
            assert life['target_started'] and life['target_uid']==plan['target_uid']!=0
            program=z.read(prefix+which+'/observer.bt')
            assert program.decode()==kernel.render_program(life['target_pid']) and sha(program)==life['probe_program_sha256']
        control_events=kernel.parse_events(z.read(prefix+'control/kernel.jsonl').decode(),control_life['target_pid'],z.read(prefix+'control/observer.stderr').decode())
        control=kernel.check_control(control_events,json.loads(z.read(prefix+'control/target.stdout')),control_life['target_pid'])
        control_identity=load('native-control.json')
        assert control_identity=={'sha256_before_after':sha(z.read(prefix+'native-control')),**control}
        raw=z.read(prefix+'factory/kernel.jsonl')
        fixture=ROOT/'docs/evidence/cseed-kernel-exit-20261109/corrected-linux/kernel.jsonl.gz'
        assert gzip.decompress(fixture.read_bytes())==raw
        started=time.monotonic()
        events=kernel.parse_events(raw.decode(),factory_life['target_pid'],z.read(prefix+'factory/observer.stderr').decode())
        replay_seconds=time.monotonic()-started
        assert len(events)==284752
        roots=[n.removesuffix('inputs.json') for n in names if n.startswith('.test_output/bstring-source-compiler/') and n.endswith('/inputs.json')]
        assert len(roots)==1
        factory=roots[0];identity=json.loads(z.read(factory+'inputs.json'))
        assert identity['before']==identity['after'] and len(identity['before'])==67
        assert identity['product_before_run']==identity['product_after_run']==sha(z.read(factory+'compiler'))
        commands=[json.loads(line) for line in z.read(factory+'commands.jsonl').splitlines()]
        assert len(commands)==1 and commands[0]['returncode']==0 and commands[0]['label']=='source-build'
        argv=commands[0]['argv'];root=Path(argv[0]).parents[1]
        assert argv==[str(root/'bin/nanoc_c'),str(root/'src_nano/nanoc_v06.nano'),'--keep-c','-o',str(root/factory/'compiler')]
        for path,expected in identity['before'].items():
            relative=Path(path).relative_to(root).as_posix()
            actual=providers['bin/nanoc_c'] if relative=='bin/nanoc_c' else sha(git_bytes(PIN,relative))
            assert expected==actual,path
        stderr=z.read(factory+'source-build.stderr').decode()
        phases=shadow_phases(stderr,True,10);assert phases['valid']
        trace=shadow_trace(stderr,root)
        assert trace['starts']==trace['completes']==1286 and not trace['failures']
        status=phases['events'][-1];parent=status['pid'];child=status['child_pid']
        assert status['completed'] and not status['parent_timeout'] and status['wait_status']==0
        assert any(e['kind']=='fork' and e['pid']==parent and e['value']==child for e in events)
        intervals=kernel.exit_intervals(events,child);assert len(intervals)==1
        waits=[e for e in events if e['tid']==parent and e['kind']=='wait-return' and e['value']==child];assert len(waits)==1
        origin=status['parent_start_seconds']*1000000000+status['parent_start_nanoseconds']
        child_events=[{**e,'seconds_from_original_parent_start':(e['monotonic_ns']-origin)/1e9} for e in events if e['tid']==child]
        relevant=[e for e in child_events if e['kind'] in ('exit-group','exit-thread','process-exit','exit-mmap-enter','exit-mmap-return','task-free')]
        reap=(waits[0]['monotonic_ns']-origin)/1e9
        fork_counts={str(pid):sum(e['kind']=='fork' and e['pid']==pid for e in events) for pid in (factory_life['target_pid'],parent,child)}
        assert len(images)==8 and len(objects)==1
    output.mkdir(parents=True);retained.mkdir(parents=True)
    for name,data in saved.items():
        p=retained/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p.read_bytes())==sha(data)
    for name,value in [('inventory',inventory),('texts',texts)]:
        with gzip.open(output/(name+'.json.gz'),'wt') as stream:json.dump(value,stream,sort_keys=True)
    report={'run_id':42588515731,'artifact_id':13661595711,'source_pin':PIN,'archive_sha256':sha(archive.read_bytes()),'entries_crc_checked':len(inventory),'diagnostic_job_conclusion':'failure: quadratic collector reached its twenty-minute step limit','factory_lifecycle':factory_life,'control':control,'kernel_events':len(events),'offline_replay_seconds':replay_seconds,'new_factory_attempts':0,'original_factory_attempts':1,'creation_succeeded':True,'native_compiler_created':True,'new_compiler_execution_qualified':False,'shared_git_matched_inputs':197,'observer_git_matched_inputs':4,'stable_factory_inputs':67,'uploaded_providers_match_recorded_before':6,'source_trace':trace,'source_phases':phases,'child_exit_mmap':intervals[0],'child_kernel_boundaries':relevant,'kernel_reap_seconds_from_original_parent_start':reap,'complete_child_reaped_within_ten_seconds':0<=reap<=10,'forks_by_known_role_pid':fork_counts,'all_images':images,'relocatable_objects':objects,'observer_and_host_compiler_final_hash_reports_reached':False,'historical_failure_cause_established':False,'whole_factory_gain_established':False,'native_sdk_closure_qualified':False,'release_qualified':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('creation_succeeded','kernel_events','offline_replay_seconds','child_exit_mmap','kernel_reap_seconds_from_original_parent_start','complete_child_reaped_within_ten_seconds','forks_by_known_role_pid')}))
    print(json.dumps(relevant))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('output',type=Path);parser.add_argument('retained',type=Path)
    args=parser.parse_args();audit(args.archive,args.output,args.retained)
