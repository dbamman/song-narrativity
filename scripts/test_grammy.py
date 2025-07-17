import sys
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from math import sqrt

labs={}

anns={}


def read_grammy(filename):

	for genre in ["all", "country", "r_and_b", "rock", "rap"]:
		yes_vals=[]
		no_vals=[]
		with open(filename) as file:
			for line in file:
				cols=line.rstrip().split("\t")

				yes_url=cols[4]
				no_url=cols[6]

				cat=cols[7]

				if cat != genre and genre != "all":
					continue

				if not yes_url.startswith("http") or not no_url.startswith("http"):
					continue

				if yes_url == no_url:
					print(yes_url)
							
				assert yes_url != no_url

				labs[yes_url]=1
				labs[no_url]=0


				yes_vals.append(anns[yes_url])
				no_vals.append(anns[no_url])

		testval=ttest_rel(yes_vals, no_vals)

		nom_val=np.mean(yes_vals)
		non_nom_val=np.mean(no_vals)
		lift=100*((nom_val-non_nom_val)/nom_val)
		n=len(yes_vals)
		pval=testval.pvalue
		if genre == "r_and_b":
			genre="R\&B"
		print("%s&%.2f&%.2f&%.1f&%s&%.3f \\\\" % (genre, nom_val, non_nom_val, lift, n, pval))


def proc(filename):

	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")	
			url=cols[0]

			label=None
			

			mean=float(cols[1])
			

			anns[url]=mean


proc(sys.argv[2])
read_grammy(sys.argv[1])
