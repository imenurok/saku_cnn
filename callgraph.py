import subprocess
import argparse

linelist=[
"less log|grep iteration > log_pretty",
"python plot.py log_pretty"
]

for line in linelist:
	subprocess.call(line,shell=True)
