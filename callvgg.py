import subprocess
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('gpu_no', help='GPU ID (negative value indicates CPU)')
parser.add_argument('--test_no', '-t',type=int , default=0, help='set area of test file for 0 to 9')
parser.add_argument('--epoch', '-E', default=100, help='Number of epochs to learn')
args = parser.parse_args()

if args.test_no<0 or args.test_no>9:
    print "error"

else:
    linelist=[
    "rm -rf images",
    "rm -rf images_raw",
    "python make_train_data.py -t "+str(args.test_no),
    "mv images images_raw",
    "mkdir images",
    "python crop.py images_raw images",
    "python compute_mean.py train.txt",
    "python train_imagenet.py -g "+args.gpu_no+" -B 16 -b 1 -E "+args.epoch+" train.txt test.txt 2>&1 | tee log"
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