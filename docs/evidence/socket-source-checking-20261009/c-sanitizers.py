import os,shlex,subprocess,tempfile,unittest,sys
from pathlib import Path
root=Path('/Users/jordanh/Src/nanolang');os.chdir(root);sys.path.insert(0,str(root))
for target,output in [('test-service-bodies-sanitize','obj/test_service_bodies_sanitize'),('test-service-ownership-sanitize','obj/test_service_ownership_sanitize')]:
 result=subprocess.run(['make','-f','Makefile.gnu','-n','CC=/opt/homebrew/opt/llvm/bin/clang',target],capture_output=True,text=True,check=True)
 commands=[shlex.split(line) for line in result.stdout.splitlines() if line.startswith('/opt/homebrew/opt/llvm/bin/clang ') and ('-o '+output+' ') in line]
 assert len(commands)==1,(target,len(commands))
 print('BUILD',shlex.join(commands[0]),flush=True)
 subprocess.run(commands[0],check=True)
os.environ['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1'
os.environ['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1'
os.environ['NANO_SERVICE_BODY_C_RUNNER']=str(root/'obj/test_service_bodies_sanitize')
os.environ['NANO_SERVICE_OWNERSHIP_C_RUNNER']=str(root/'obj/test_service_ownership_sanitize')
from tests.test_socket_service_source import SocketBodies,SocketOwnership
class COnly:
 @classmethod
 def setUpClass(cls):
  cls.directory=tempfile.TemporaryDirectory(prefix='nl51-tcp-sanitizer-')
  cls.addClassCleanup(cls.directory.cleanup);cls.work=Path(cls.directory.name).resolve()
  (cls.work/'interface.nsi.json').write_bytes((root/'tests/fixtures/nsi_socket_plan.json').read_bytes())
  cls.probes=[];cls.drivers=[]
class CSanitizedBodies(COnly,SocketBodies):pass
class CSanitizedOwnership(COnly,SocketOwnership):
 def test_lowering_does_not_reinterpret_tcp_as_file(self):
  self.skipTest('I qualify publication through the actual drivers in the paired batch.')
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in [CSanitizedBodies,CSanitizedOwnership]])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
