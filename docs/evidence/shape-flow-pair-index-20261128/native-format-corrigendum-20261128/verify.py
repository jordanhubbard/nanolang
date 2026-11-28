"""I classify actual native bytes while retaining a mislabeled legacy report."""
from pathlib import Path
import argparse,gzip,hashlib,json,subprocess,tempfile,zipfile
BASE='docs/evidence/shape-flow-pair-index-20261128/original-cohorts-46119624230/linux-bootstrap'
REF='cef08089de38e5e1e50c69936614346629569965'
sha=lambda d:hashlib.sha256(d).hexdigest()
def classify(image):
    if image[:6]==b'\x7fELF\x02\x01' and len(image)>=20:
        machine=int.from_bytes(image[18:20],'little')
        return dict(format='ELF64 little-endian',architecture={62:'x86-64',183:'AArch64'}.get(machine,'unknown'),elf_e_machine=machine,macho_arm64=False)
    if image[:4]==bytes.fromhex('cffaedfe') and len(image)>=8 and int.from_bytes(image[4:8],'little')==0x100000c:
        return dict(format='Mach-O',architecture='ARM64',macho_arm64=True)
    raise ValueError('I require a recognized retained native header.')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--observations',type=Path,default=Path(__file__).with_name('observations.json'));a=p.parse_args()
    raw=a.observations.read_bytes();assert sha(raw)=='39524eb53e57c4bd6027d5d0e522c51f1e1033ee49abfa76e9b89490bbb81440';observed=json.loads(raw)
    assert not observed['product_execution_repeated'] and not observed['product_gate_acceptance_changed'] and not observed['whole_release']
    assert observed['product']=='ddd3b5a4ff35678d1906c07c08fc6bbc39b041de'
    read=lambda path:subprocess.check_output(['git','show',REF+':'+BASE+'/'+path],cwd=a.repo)
    data=read('manifest.json');assert sha(data)=='712a95ad550e72c0988aad66025bb47f9ee420dce1c68f3219331298ef272ddc';manifest=json.loads(data)
    with tempfile.TemporaryDirectory(prefix='native-format-reader-',dir=a.workspace) as d:
        root=Path(d)
        for name in ['linux-bootstrap.zip','verified-linux-bootstrap.json']:
            record=next(r for r in manifest['files'] if r['path']==name);target=root/name;h=hashlib.sha256();size=0
            with target.open('wb') as f:
                for key in record['chunks']:
                    packed=read('objects/'+key+'.gz');meta=manifest['chunks'][key];assert sha(packed)==meta['gzip_sha256'];block=gzip.decompress(packed);assert sha(block)==key and len(block)==meta['bytes'];f.write(block);h.update(block);size+=len(block)
            assert size==record['size'] and h.hexdigest()==record['sha256']
        legacy=json.loads((root/'verified-linux-bootstrap.json').read_text())[0];assert legacy['native_macho_arm64'] is True
        assert sha((root/'linux-bootstrap.zip').read_bytes())==observed['archive_sha256'] and (root/'linux-bootstrap.zip').stat().st_size==observed['archive_bytes']==125166776
        with zipfile.ZipFile(root/'linux-bootstrap.zip') as z:
            assert z.testzip() is None;name=next(n for n in z.namelist() if n.endswith('/bin/nanoc_bootstrap.json'));receipt=json.loads(z.read(name));image=z.read(name.removesuffix('bin/nanoc_bootstrap.json')+'bin/nanoc_stage2')
            assert sha(image)==receipt['artifacts']['nanoc_stage2']==observed['actual_stage2_sha256']
            actual=classify(image);assert actual==dict(format=observed['actual_native_format'],architecture=observed['actual_native_architecture'],elf_e_machine=observed['actual_elf_e_machine'],macho_arm64=observed['actual_native_macho_arm64'])
            assert image[:20].hex()==observed['actual_header_hex'] and not actual['macho_arm64'] and actual['elf_e_machine']==62
    print('I verify actual ELF64/X86-64 Stage2 bytes and preserve the legacy mislabeled native_macho_arm64=true report. Product acceptance and release status remain unchanged.')
if __name__=='__main__':main()
