import sys, re
from collections import Counter
import random
import numpy as np

random.seed(1)

def bootstrap(vals, B=1000):
	measures=[]
	for b in range(B):
		resample=random.choices(vals, k=len(vals))
		measures.append(np.mean(resample))
	return np.percentile(measures, [2.5, 50, 97.5])

hiphop_rap=set(["Hip hop music", "Pop rap", "Trap music", "Pop-rap", "Dirty rap", "Rap rock", "Trap music (hip hop)", "G-funk", "Gangsta rap", "Hip house", "Emo rap", "Trap music (EDM)", "Hip hop soul", "Trap music (hip hop)", "Southern hip hop", "East Coast hip hop", "Alternative hip hop", "Hip hop", "West Coast hip hop", "Hardcore hip hop", "Conscious hip hop", "Comedy hip hop", "Hip-hop music"])
country=set(["Country music", "Country pop", "Country rock", "Countrypolitan", "Country folk", "Country blues", "Bro-country"])

def read_top(filename):
	
	hiphop_rap_counts={}
	country_counts={}
	
	with open(filename) as file:
		file.readline()
		for line in file:
			cols=line.split("\t")
			year=int(cols[0])

			if cols[5] == "NONE" or cols[5] == "":
				continue

			genres=cols[5].rstrip().split("#")

			hiphop_rap_genre_match=0
			country_match=0

			for gen in genres:
				g=re.sub("/wiki/", "", gen)
				g=re.sub("_", " ", g)

				if g in hiphop_rap:
					hiphop_rap_genre_match=1
				if g in country:
					country_match=1

			if year not in hiphop_rap_counts:
				hiphop_rap_counts[year]=[]
				country_counts[year]=[]

			if hiphop_rap_genre_match:
				hiphop_rap_counts[year].append(1)
			else:
				hiphop_rap_counts[year].append(0)
				
			if country_match:
				country_counts[year].append(1)
			else:
				country_counts[year].append(0)

				

	for year in hiphop_rap_counts:
		low, mid, high=bootstrap(hiphop_rap_counts[year])
		print("hiphop\t%s\t%.3f\t%.3f\t%.3f" % (year, mid, low, high))
	for year in country_counts:
		low, mid, high=bootstrap(country_counts[year])
		print("country\t%s\t%.3f\t%.3f\t%.3f" % (year, mid, low, high))

read_top(sys.argv[1])