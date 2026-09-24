#!/usr/bin/env python3
import os,sys,shutil
shutil.copyfile(sys.argv[2],'/tmp/pr522-ci-toolchain/nested-shadows.nvm')
os.environ['NANO_VM_TRACE']='1'
vm='/Users/jordanh/Src/nanolang-pr522-repair/bin/nano_vm'
os.execv(vm,[vm,*sys.argv[1:]])
