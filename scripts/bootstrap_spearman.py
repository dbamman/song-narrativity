import sys
import random
from scipy.stats import spearmanr
import numpy as np

B=10000

with open(sys.argv[1]) as file:
	vals=[]
	for line in file:
		cols=line.rstrip().split("\t")
		cols[0]=float(cols[0])
		cols[1]=float(cols[1])
		vals.append((cols[0], cols[1]))

	resample_vals=[]
	for b in range(B):
		resample=random.choices(vals, k=len(vals))
		p=[]
		g=[]
		for one, two in resample:
			p.append(one)
			g.append(two)

		val, _ = spearmanr(p, g)
		resample_vals.append(val)


	percs=np.percentile(resample_vals, [2.5, 50, 97.5])
	print("%.3f [%.3f-%.3f]" % (percs[1], percs[0], percs[2]))