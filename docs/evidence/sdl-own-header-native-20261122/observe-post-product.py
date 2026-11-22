from pathlib import Path
import hashlib,json,subprocess,datetime
root=Path('/home/jkh/.worktrees/nanolang/own-header-1181-puck-20261122');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_pin':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'root_product':sha(root/'bin/nanoc_c'),'before_tests_product':sha(root/'.test_output/own-header-1181/nanoc_c'),'helper_after':sha(root/'bin/nano_as_capture.so')};r['product_stable_after_original_target']=r['root_product']==r['before_tests_product'];print(json.dumps(r,indent=2))
