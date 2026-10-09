"""I run the independent Nano compiler driver in VM and native form."""
import os
from pathlib import Path
import shlex
from tests.test_service_drivers import ServiceDrivers, ROOT

class NanoServiceDriver(ServiceDrivers):
    def setUp(self):
        super().setUp()
        module=os.environ.get('NANO_FILE_DRIVER_MODULE')
        native=os.environ.get('NANO_FILE_DRIVER_NATIVE')
        if not module or not native:self.skipTest('I require the qualified Nano driver module and native product')
        self.env['NANOLANG_ROOT']=str(ROOT)
        self.drivers=[]
        tools=self.work/'tools';tools.mkdir()
        for name,command in [('nano_driver_vm',[ROOT/'bin/nano_vm',module,'--']),('nano_driver_native',[native])]:
            wrapper=tools/name
            wrapper.write_text('#!/bin/sh\nexec '+shlex.join(list(map(str,command)))+' "$@"\n')
            wrapper.chmod(0o700)
            self.drivers.append(wrapper)
