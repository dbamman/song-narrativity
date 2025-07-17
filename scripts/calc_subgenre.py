import sys, re
import numpy as np
import random

random.seed(1)

vals={}

def bootstrap(vals, B=1000):
	measures=[]
	for b in range(B):
		resample=random.choices(vals, k=len(vals))
		measures.append(np.mean(resample))
	return np.percentile(measures, [2.5, 50, 97.5])

preds={}

def proc(filename):

	with open(filename) as file:

		for line in file:
			cols=line.rstrip().split("\t")

			url=cols[0]
			val=float(cols[1])
			preds[url]=val



def read_top(filename):
	subs={}
	with open(filename) as file:
		file.readline()
		for line in file:
			cols=line.rstrip().split("\t")
			subchart=cols[1]
			year=int(cols[0])
			
			if subchart not in subs:
				subs[subchart]={}
			url=cols[5]

			if not url.startswith("http"):
				continue
	
			pred=preds[url]

			if year not in subs[subchart]:
				subs[subchart][year]=[]
			subs[subchart][year].append(pred)
		

	for subchart in subs:
		for year in subs[subchart]:
			lower, mid, upper=bootstrap(subs[subchart][year])
			n=len(subs[subchart][year])
			if n > 0:
				print("%s\t%.3f\t%s\t%s\t%s\t%s\tYEAR"%  (subchart, mid, year, lower, upper, n))

	
proc(sys.argv[1])
read_top(sys.argv[2])
