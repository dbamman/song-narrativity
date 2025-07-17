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
			agent=float(cols[2])
			events=float(cols[3])
			world=float(cols[4])

			preds[url]=val, agent, events, world


def read_top(filename):
	subs={}

	genres_vals={"ALL": {}, "AGENT":{}, "EVENT":{}, "WORLD": {}}

	with open(filename) as file:
		file.readline()
		for line in file:
			cols=line.split("\t")

			genres=cols[5].rstrip().split("#")

			newg=[]
			for gen in genres:
				g=re.sub("/wiki/", "", gen)
				g=re.sub("_", " ", g)

				newg.append(g)

			genres=newg
			
			year=int(cols[0])

			for genre in genres:
				if genre not in subs:
					subs[genre]=[]

			for gen in genres:
				if gen not in genres_vals["ALL"]:
					genres_vals["ALL"][gen]=[]
					genres_vals["AGENT"][gen]=[]
					genres_vals["EVENT"][gen]=[]
					genres_vals["WORLD"][gen]=[]

			url=cols[3]

			if not url.startswith("http"):
				continue
	
			if url in preds:
				val, agent, events, world=preds[url]
				
				for gen in genres:
					genres_vals["ALL"][gen].append(val)
					genres_vals["AGENT"][gen].append(agent)
					genres_vals["EVENT"][gen].append(events)
					genres_vals["WORLD"][gen].append(world)


			else:
				print("Missing", url)


	sortedvals=[]
	for gen in genres_vals["ALL"]:
		if gen == "NONE":
			continue
		if len(genres_vals["ALL"][gen]) >= 200:
			lower, mid, upper=bootstrap(genres_vals["ALL"][gen])
			a_lower, a_mid, a_upper=bootstrap(genres_vals["AGENT"][gen])
			e_lower, e_mid, e_upper=bootstrap(genres_vals["EVENT"][gen])
			w_lower, w_mid, w_upper=bootstrap(genres_vals["WORLD"][gen])

			gen=re.sub(" music", "", gen)
			gen=re.sub("Contemporary R%26B", "Contemporary R\\&B", gen)
			sortedvals.append((mid, "%s&\t%.2f {\\small[%.2f-%.2f]}&%.2f {\\small[%.2f-%.2f]}&%.2f {\\small[%.2f-%.2f]}&%.2f {\\small[%.2f-%.2f]}\\\\" % (gen, mid, lower, upper, a_mid, a_lower, a_upper, e_mid, e_lower, e_upper, w_mid, w_lower, w_upper)))

	sortedvals=sorted(sortedvals, reverse=True)
	for val in sortedvals:
		print(val[1])
	
proc(sys.argv[1])
read_top(sys.argv[2])
