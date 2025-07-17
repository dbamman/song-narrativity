import sys
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from math import sqrt

labs={}

anns={}


def read_grammy(filename):

	yes_vals=[]
	no_vals=[]
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")

			if cols[0] == "Grammy_NOM":
				yes_vals.append(float(cols[9]))

			elif cols[0] == "Grammy_ALT":
				no_vals.append(float(cols[9]))


		assert len(yes_vals) == len(no_vals)

		testval=ttest_rel(yes_vals, no_vals)

		nom_val=np.mean(yes_vals)
		non_nom_val=np.mean(no_vals)
		lift=100*((nom_val-non_nom_val)/nom_val)
		n=len(yes_vals)
		pval=testval.pvalue

		print("%s\t%.2f\t%.2f\t%.1f\t%s\t%.3f" % ("annotated", nom_val, non_nom_val, lift, n, pval))



read_grammy(sys.argv[1])
