"""I audit the fixed actual GCC observation without rerunning a factory."""
import argparse
import functools
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
PIN = '35a32bf746e9c7741158bb8e62563d92bde614f8'
BASE = '3c3f1edae92125dd44cf392ddc0f160a2868e129'

def digest(data):
    return hashlib.sha256(data).hexdigest()

@functools.cache
def git_digest(pin, path):
    return digest(subprocess.check_output(['git','show',pin+':'+path],cwd=ROOT))

def audit(archive, output, retained):
    assert not output.exists() and not retained.exists()
    parser = ROOT/'scripts/compare_cseed_source_creation.py'
    assert digest(parser.read_bytes()) == git_digest(PIN,'scripts/compare_cseed_source_creation.py')
    spec=importlib.util.spec_from_file_location('comparison',parser)
    comparison=importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)
    inventory, texts, native, saved, findings = [], {}, [], {}, []
    with zipfile.ZipFile(archive) as z:
        names=z.namelist(); assert len(names)==len(set(names))
        for info in z.infolist():
            assert not info.filename.startswith('/') and '..' not in Path(info.filename).parts
            if info.is_dir(): continue
            data=z.read(info)
            inventory.append({'path':info.filename,'bytes':len(data),'sha256':digest(data)})
            if data[:4]==b'\x7fELF':
                assert data[5] in (1,2)
                kind=struct.unpack_from('<H' if data[5]==1 else '>H',data,16)[0]
                assert kind in (2,3)
                native.append({'path':info.filename,'bytes':len(data),'sha256':digest(data),'kind':'ELF executable/PIE'})
                saved[info.filename]=data
            else:
                texts[info.filename]=data.decode()
        prefix='cseed-source-comparison/'
        load=lambda name:json.loads(z.read(prefix+name))
        plan=load('plan.json'); summary=load('summary.json')
        assert plan['baseline']==BASE and plan['candidate']==PIN
        assert plan['order']==['original','modified','modified','original','original','modified']
        assert plan['shadow_timeout_seconds']==10 and plan['shadow_phase_observation']['environment']=='NANO_SHADOW_PHASE_TRACE=1'
        assert summary['comparison_valid'] and summary['all_planned_samples_finished'] and summary['phase_observation_valid']
        assert not summary['all_creations_passed'] and len(summary['samples'])==6
        providers=load('providers-before.json'); assert providers==load('providers-after.json')
        for variant in ('original','modified'):
            assert digest(z.read(prefix+'providers/'+variant+'/nanoc_c'))==providers[variant]['bin/nanoc_c']
        shared=load('shared-inputs.json'); assert len(shared)==197
        for path,value in shared.items():
            assert git_digest(BASE,path)==git_digest(PIN,path)==value,path
        observed_prefix=[]
        for sample in summary['samples']:
            index,variant=sample['index'],sample['variant']
            assert variant==plan['order'][index-1] and not sample['creation_succeeded'] and sample['process']['returncode']==1
            sample_root=prefix+'samples/'+f'{index:02}-{variant}/'
            identity=json.loads(z.read(sample_root+'inputs.json'))
            assert identity['before']==identity['after'] and len(identity['before'])==67
            assert identity['product_before_run'] is identity['product_after_run'] is None
            for path,value in identity['before'].items():
                relative=Path(path).relative_to(plan['worktrees'][variant]).as_posix()
                expected=providers[variant]['bin/nanoc_c'] if relative=='bin/nanoc_c' else git_digest(BASE if variant=='original' else PIN,relative)
                assert value==expected,path
            raw=z.read(sample_root+'source-build.stderr').decode()
            trace=json.loads(json.dumps(comparison.shadow_trace(raw,Path(plan['worktrees'][variant]))))
            assert trace==sample['trace'] and trace['failures']==0
            selection=trace['selection_prefix']; common=min(len(selection),len(observed_prefix))
            assert selection[:common]==observed_prefix[:common]
            if len(selection)>len(observed_prefix): observed_prefix=selection
            finding={'index':index,'variant':variant,'starts':trace['starts'],'completes':trace['completes'],'creation_succeeded':False}
            if variant=='modified':
                phases=comparison.shadow_phases(raw,False,10)
                assert phases==sample['phase_observation'] and phases['valid']
                status=phases['events'][-1]
                start=status['parent_start_seconds']+status['parent_start_nanoseconds']/1e9
                elapsed=lambda event:event['monotonic_seconds']+event['monotonic_nanoseconds']/1e9-start
                finding['phase_seconds_from_original_parent_start']=[{'phase':e['phase'],'seconds':elapsed(e)} for e in phases['events']]
                finding['supervisor']=status
                assert status['parent_timeout'] and not status['clock_failure']
                if index in (3,6):
                    assert status['completed'] and status['wait_status']==0 and status['exit_code']==0
                    flush=next(e for e in phases['events'] if e['phase']=='flush-end')
                    assert elapsed(flush)<10<elapsed(status)
                    finding['actual_child_exit_time_established']=False
                else:
                    assert index==2 and status['signal']==14 and not status['completed']
            findings.append(finding)
        assert len(observed_prefix)==1286
        controls_root=next(p.removesuffix('operations.json') for p in names if p.startswith('shadow-completion-') and p.endswith('/operations.json'))
        controls=json.loads(z.read(controls_root+'operations.json')); assert len(controls)==9
        for op in controls:
            label=op['label']
            if label=='native-errno':
                assert op['returncode']==op['build_returncode']==0
                assert digest(z.read(controls_root+'errno-native'))==op['native_sha256_before_after']
            else:
                assert digest(z.read(controls_root+label+'/input.nano'))==op['input_sha256_before_after']
                assert op['returncode']==(0 if label in ('disabled','enabled','phase-audit') else 1)
        assert len(native)==6
    output.mkdir(parents=True);retained.mkdir(parents=True)
    for path,data in saved.items():
        target=retained/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        assert digest(target.read_bytes())==digest(data)
    for name,value in [('inventory',inventory),('texts',texts)]:
        with gzip.open(output/(name+'.json.gz'),'wt') as f:json.dump(value,f,sort_keys=True)
    report={'run_id':42580473796,'artifact_id':13657921443,'source_pin':PIN,'baseline':BASE,'archive_sha256':digest(archive.read_bytes()),'all_entries_crc_checked':len(inventory),'shared_git_matched_inputs':197,'factory_stable_inputs_each':67,'all_six_attempts_failed':True,'modified_phase_audits_valid':True,'samples':findings,'native_products':native,'control_operations':controls,'source_compiler_products':0,'actual_child_exit_time_established':False,'historical_failure_cause_established':False,'whole_factory_performance_improvement_established':False,'release_qualified':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'entries':len(inventory),'native_images':len(native),'observed_modified_attempts':3,'all_six_creations_failed':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path);p.add_argument('retained',type=Path)
    a=p.parse_args();audit(a.archive,a.output,a.retained)
