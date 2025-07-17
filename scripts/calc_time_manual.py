import sys, re
import numpy as np
import random

random.seed(1)

def bootstrap(vals, B=1000):
	measures=[]
	for b in range(B):
		resample=random.choices(vals, k=len(vals))
		measures.append(np.mean(resample))
	return np.percentile(measures, [2.5, 50, 97.5])

def read_annotations(filename):
	years={}
	with open(filename) as file:
		file.readline()
		for line in file:
			cols=line.rstrip().split("\t")

			if cols[0] != "Billboard":
				continue

			year=int(cols[1])

			url=cols[5]

			pred=float(cols[9])

			if year not in years:
				years[year]=[]
			years[year].append(pred)
			

	for year in years:
		lower, mid, upper=bootstrap(years[year])
		n=len(years[year])
		if n > 0:
			print("%.3f\t%s\t%s\t%s\t%s\tYEAR"%  (mid, year, lower, upper, n))

	
read_annotations(sys.argv[1])
