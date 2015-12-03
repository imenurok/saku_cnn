import sys
import argparse
import commands
import subprocess
import os

parser = argparse.ArgumentParser(
    description='Image inspection using chainer')
parser.add_argument('--source','-s',default='Picture', help='Path to inspection image file')
args = parser.parse_args()

def cmd(cmd):
	p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	p.wait()
	stdout, stderr = p.communicate()
	return stdout.rstrip()

#make directries
cmd("mkdir images")

#copy images and make train.txt
imageDir = "images"
train = open('train.txt','w')
test = open('test.txt','w')
labelsTxt = open('labels.txt','w')

labels=os.listdir(args.source)
classNo=0
cnt = 0
#label = labels[classNo]
for label in labels:
	workdir = args.source+"/"+label
	labelsTxt.write(label+"\n")
	imageCnt=0
	images=os.listdir(workdir)
	startCnt=cnt
	length = len(images)
	for image in images:
		imagepath = imageDir+"/image%07d" %cnt +".jpg"
		cmd("cp "+workdir+"/"+image+" "+imagepath)
		if cnt-startCnt < length*0.75:
			train.write(imagepath+" %d\n" % classNo)
		else:
			test.write(imagepath+" %d\n" % classNo)
		print imagepath 
		cnt += 1
	
	classNo += 1

train.close()
test.close()
labelsTxt.close()
