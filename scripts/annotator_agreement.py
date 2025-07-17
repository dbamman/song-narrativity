import sys
import numpy as np
from scipy.stats import spearmanr

from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import interval_distance 

def krippendorff_alpha(annotation_triples):

    t = AnnotationTask(annotation_triples, distance=interval_distance)
    result = t.alpha()
    return result


def proc(filename):

	anns=[]
	ads=[]

	a1s=[]
	a2s=[]
	a3s=[]
	with open(filename) as file:
		file.readline()
		for line in file:
			cols=line.rstrip().split("\t")
			url=cols[3]
			a1_avg=float(cols[22])
			a2_avg=float(cols[26])
			a3_avg=float(cols[30])

			print(a1_avg, a2_avg, a3_avg)

			a1s.append(a1_avg)
			a2s.append(a2_avg)
			a3s.append(a3_avg)

			mean=(a1_avg + a2_avg + a3_avg)/3
			ad=(abs(a1_avg-mean) + abs(a2_avg-mean) + abs(a3_avg-mean) ) / 3
			ads.append(ad)

			anns.append(("a1", url, a1_avg))
			anns.append(("a2", url, a2_avg))
			anns.append(("a3", url, a3_avg))						

	print("Krippendorff alpha: %.3f" % krippendorff_alpha(anns))
	print("AD: %.3f" % np.mean(ads))

	s12,_=spearmanr(a1s,a2s)
	s23,_=spearmanr(a2s,a3s)
	s13,_=spearmanr(a1s,a3s)

	avgsp=(s12+s23+s13)/3
	print("Average spearman: %.3f" % avgsp, s12, s23, s13)



proc(sys.argv[1])	