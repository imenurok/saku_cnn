import subprocess
import argparse

linelist=[
"less log|grep iteration > log_pretty",
"python plot.py log_pretty"
]

interrupt_check=False
for line in linelist:
    if interrupt_check==False:
        p = subprocess.Popen(line, shell=True)
        try:
            p.wait()
        except KeyboardInterrupt:
            try:
               p.communicate()[0]
            except OSError:
               pass
            interrupt_check=True