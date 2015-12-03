import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("gpu_no")
args = parser.parse_args()

linelist=[
"rm -rf images",
"rm -rf images_raw",
"python make_train_data.py",
"mv images images_raw",
"mkdir images",
"python crop.py images_raw images",
"python compute_mean.py train.txt",
"python train_imagenet.py -g "+args.gpu_no+" -B 16 -b 1 -E 100 train.txt test.txt 2>&1 | tee log"
]

for line in linelist:
	subprocess.call(line,shell=True)
