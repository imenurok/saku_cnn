import matplotlib
matplotlib.use("Agg")

import pylab
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("result")
args = parser.parse_args()

xs_train = [0]
ys_train = [1]

xs_test = [0]
ys_test = [1]

for line in open(args.result):
    print line
    data = json.loads(line)
    if int(data["iteration"])<=130000:
        if data["type"]=="train":
            xs_train.append(data["iteration"])
            ys_train.append(data["error"])
        elif data["type"]=="val":
            xs_test.append(data["iteration"])
            ys_test.append(data["error"])
pylab.xlabel("iteration")
pylab.ylabel("error")
pylab.plot(xs_train, ys_train)
pylab.plot(xs_test,ys_test)
pylab.xlim(0,xs_test[len(xs_test)-1])
pylab.ylim(0,1)
pylab.savefig("graph.png")