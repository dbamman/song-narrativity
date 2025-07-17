import sys
from scipy.stats import spearmanr
import numpy as np

meta={}
all_preds={}

def read_preds(filename):
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")
			url=cols[0]
			pred=float(cols[1])
			all_preds[url]=pred

def read_meta(filename):

	preds={}

	with open(filename) as file:
		file.readline()
		for line in file:
			cols=line.rstrip().split("\t")
			year=int(cols[0])
			url=cols[3].lstrip().rstrip()
			if url == "NONE":
				continue

			if year not in preds:
				preds[year]=[]

			preds[year].append(all_preds[url])

	year_val=[]
	year_pred=[]

	for year in preds:
		mean=np.mean(preds[year])
		year_pred.append(mean)
		year_val.append(year)
		print("%s\t%s" % (year, mean))
	
	print(spearmanr(year_val, year_pred))
	print("1960: %.3f, 2024: %.3f, diff: %.3f" % (np.mean(preds[1960]), np.mean(preds[2024]), np.mean(preds[2024])-np.mean(preds[1960])))

read_preds(sys.argv[1])			
read_meta(sys.argv[2])

			