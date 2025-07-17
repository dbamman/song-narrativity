import sys, re
import numpy as np
import random

random.seed(1)

vals={}

def convert_url_to_path(url, top):
	parts=url.split("/")
	if len(parts) < 2:
		return None
	# print(parts)
	path="%s/%s_%s" % (top, parts[-2], parts[-1])
	return path

def bootstrap(vals, B=1000):
	measures=[]
	for b in range(B):
		resample=random.choices(vals, k=len(vals))
		measures.append(np.mean(resample))
	return np.percentile(measures, [2.5, 50, 97.5])

preds={}

def read_preds(filename):

	with open(filename) as file:

		for line in file:
			cols=line.rstrip().split("\t")

			url=cols[0]
			val=float(cols[1])

			preds[url]=val


def read_billboard_hot_100(filename):
	years={}
	with open(filename) as file:
		file.readline()
		for line in file:
			cols=line.rstrip().split("\t")
			year=int(cols[0])

			url=cols[3]

			if not url.startswith("http"):
				continue

			pred=preds[url]

			if year not in years:
				years[year]=[]
			years[year].append(pred)
			

	for year in years:
		lower, mid, upper=bootstrap(years[year])
		n=len(years[year])
		if n > 0:
			print("%.3f\t%s\t%s\t%s\t%s\tYEAR"%  (mid, year, lower, upper, n))

	
read_preds(sys.argv[1])
read_billboard_hot_100(sys.argv[2])
