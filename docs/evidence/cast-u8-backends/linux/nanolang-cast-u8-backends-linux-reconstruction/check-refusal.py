import pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1]);p.write_bytes(b'preserve-output-sentinel\n');r=subprocess.run(sys.argv[2:],stdout=subprocess.PIPE,stderr=subprocess.PIPE);sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);assert r.returncode!=0,r.returncode;assert p.read_bytes()==b'preserve-output-sentinel\n';print('I refused conversion and preserved existing output.')
