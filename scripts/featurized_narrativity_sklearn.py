import sklearn, optuna, argparse
import torch
import torch.nn as nn
import numpy as np
import random
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
import sys, json
from math import log
import os
from collections import Counter
from scipy import sparse
import operator
from sklearn.preprocessing import StandardScaler

from booknlp.booknlp import BookNLP
import sys, re

model_params={
		"pipeline":"entity,supersense", 
		"model":"big"
	}
	
bestDevScore=-100
best1=best2=best3=best_reg=None

booknlp=None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

splits={}

concreteness={}

def read_concreteness(filename):
	with open(filename) as file:
		file.readline()
		for line in file:
			cols=line.rstrip().split("\t")
			term=cols[0]
			val=float(cols[2])
			concreteness[term]=val



invalid_tags=set(["#", "$", "''", "``", "(", ")", ",", ":"])


def featurize(text, feature_set, idd=None):

	global booknlp
	minimal_feats=True

	pos_counts=Counter()
	
	book_id=idd
	output_directory="%s/%s" % (booknlp_output_folder, book_id)

	if not os.path.exists("%s/%s.entities" % (output_directory, book_id)):
		with open("book.tmp", "w") as out:
			out.write("%s\n" % text)

		print("booknlp dir", output_directory)

		if booknlp is None:
			booknlp=BookNLP("en", model_params)

		booknlp.process("book.tmp", output_directory, book_id)

	ents={}
	with open("%s/%s.entities" % (output_directory, book_id)) as file:
		file.readline()
		for line in file:
			_, start, end, prop, cat, text=line.rstrip().split("\t")
			start=int(start)
			end=int(end)

			if cat == "PER":# and prop != "PRON":
				ents[start,end]=text

	with open("%s/%s.supersense" % (output_directory, book_id)) as file:
		file.readline()
			
		for line in file:
			start, end, cat, text=line.rstrip().split("\t")
			start=int(start)
			end=int(end)

			if cat == "noun.person" or cat == "noun.animal":
				ents[start,end]=text


	tokens=[]

	animate={}
	n_tokens_no_punct=0
	doc_concrete=0
	loc_preps=0

	i_me_my=0
	i_words=set(["i", "me", "my"])

	big_feats={}

	with open("%s/%s.tokens" % (output_directory, book_id)) as file:
		file.readline()
			
		lastWord=None
		for idx, line in enumerate(file):
			paragraph_ID, sentence_ID, token_ID_within_sentence, token_ID_within_document, word, lemma, byte_onset, byte_offset, POS_tag, fine_POS_tag, dependency_relation, syntactic_head_ID, event=line.rstrip().split("\t")
			pos_counts[fine_POS_tag]+=1

			# unigram bag of words features

			if feature_set == "bow+pos+animacy+concrete+imemy" or feature_set=="bow":
				big_feats[word.lower()]=1
		
			if word.lower() in set(["in", "on", "at", "out", "up"]):
				loc_preps+=1
			if word.lower() in i_words:
				i_me_my+=1
			if fine_POS_tag not in invalid_tags:
				n_tokens_no_punct+=1
			if fine_POS_tag.startswith("N") and word.lower() in concreteness:
				doc_concrete+=concreteness[word.lower()]				
			if idx > 0 and fine_POS_tag.startswith("N"):
				bigram=("%s %s" % (tokens[-1][4], word)).lower()
				if bigram in concreteness:
					doc_concrete+=concreteness[bigram]

			tokens.append(line.rstrip().split("\t"))

			
	for start, end in ents:
		head=None
		for i in range(start, end+1):
			paragraph_ID, sentence_ID, token_ID_within_sentence, token_ID_within_document, word, lemma, byte_onset, byte_offset, POS_tag, fine_POS_tag, dependency_relation, syntactic_head_ID, event=tokens[i]
			if int(syntactic_head_ID) < start or int(syntactic_head_ID) > end:
				head=i
				break

		if head is not None:
			paragraph_ID, sentence_ID, token_ID_within_sentence, token_ID_within_document, word, lemma, byte_onset, byte_offset, POS_tag, fine_POS_tag, dependency_relation, syntactic_head_ID, event=tokens[head]
			if dependency_relation == "nsubj":
				animate[head]=1

		

	if feature_set == "minimal":
		big_feats["__VBD__"]=pos_counts['VBD']/n_tokens_no_punct
		big_feats["__NN__"]=pos_counts['NN']/n_tokens_no_punct
		big_feats["__VBZ__"]=pos_counts['VBZ']/n_tokens_no_punct
		
	else:
		if feature_set == "bow+pos+animacy+concrete+imemy" or feature_set == "pos+animacy+concrete+imemy":
			for pos in pos_counts:
				rate=pos_counts[pos]/n_tokens_no_punct
				big_feats["__%s__" % pos]=rate
	
	if feature_set == "bow+pos+animacy+concrete+imemy" or feature_set == "minimal" or feature_set == "pos+animacy+concrete+imemy":

		big_feats["__concrete__"]=doc_concrete/n_tokens_no_punct
		big_feats["__animate__"]=len(animate)/n_tokens_no_punct


	if feature_set == "bow+pos+animacy+concrete+imemy" or feature_set == "pos+animacy+concrete+imemy" or feature_set=="imemy":
		big_feats["__imemy__"]=i_me_my/n_tokens_no_punct
	# print(feats, doc_concrete, n_tokens_no_punct)
	return big_feats

def convert(x, vocab):
	new_data=sparse.lil_matrix((len(x), len(vocab)))

	for idx, feats in enumerate(x):
		for f in feats:
			if f in vocab:
				new_data[idx,vocab[f]]=feats[f]
	return new_data

def print_weights(clf, vocab, n=20):
	nz=0
	weights=clf.coef_
	for w in weights:
		if w != 0:
			nz+=1
	print("nz: %s" % nz)
	reverse_vocab=[None]*len(weights)
	for k in vocab:
		reverse_vocab[vocab[k]]=k

	for feature, weight in sorted(zip(reverse_vocab, weights), key = operator.itemgetter(1))[:n]:
		if weight != 0:
			print("%.3f\t%s" % (weight, feature))


	print()

	for feature, weight in list(reversed(sorted(zip(reverse_vocab, weights), key = operator.itemgetter(1))))[:n]:
		if weight != 0:
			print("%.3f\t%s" % (weight, feature))

		
def get_feats(x):
	feat_ids={}
	for feats in x:
		for feat in feats:
			if feat not in feat_ids:
				feat_ids[feat]=len(feat_ids)
	return feat_ids

def read_data(filename, feature_set):
	train_o=[]
	train_x=[]
	train_y1=[]
	train_y2=[]
	train_y3=[]

	dev_o=[]
	dev_x=[]
	dev_y1=[]
	dev_y2=[]
	dev_y3=[]

	test_o=[]
	test_x=[]
	test_y1=[]
	test_y2=[]
	test_y3=[]

	seen={}

	def splitter(X, y, train_idx, dev_idx, test_idx):
		X_train, y_train=X[train_idx], y[train_idx]
		X_dev, y_dev=X[dev_idx], y[dev_idx]
		X_test, y_test=X[test_idx], y[test_idx]
		return X_train, y_train, X_dev, y_dev, X_test, y_test 

	with open(filename) as file:

		for line in file:
			data=json.loads(line.rstrip())

			split=data["split"]
			url=data["url"]

			agents=data["agent"]
			events=data["events"]
			world=data["world"]

			text=data["text"]

			url_parts=url.split("/")

			idd="%s_%s" % (url_parts[-2], url_parts[-1])
			feats=featurize(data, feature_set, idd=idd)

			if split == "train":
				train_x.append(feats)
				train_y1.append(agents)
				train_y2.append(events)
				train_y3.append(world)
				train_o.append(url)
				seen[url]=1
			elif split == "dev":
				dev_x.append(feats)
				dev_y1.append(agents)
				dev_y2.append(events)
				dev_y3.append(world)
				dev_o.append(url)
				seen[url]=1

			elif split == "test":
				test_x.append(feats)
				test_y1.append(agents)
				test_y2.append(events)
				test_y3.append(world)
				test_o.append(url)
				seen[url]=1

		print("train (%s), dev (%s), test (%s)" % (len(train_x), len(dev_x), len(test_x)))



	# shuffle the data
	def shuffle_all(x, y1, y2, y3, o):
		tmp = list(zip(x, y1, y2, y3, o))
		random.shuffle(tmp)
		x, y1, y2, y3, o = zip(*tmp)
		
		return x, y1, y2, y3, o

	train_x, train_y1, train_y2, train_y3, train_o=shuffle_all(train_x, train_y1, train_y2, train_y3, train_o)
	dev_x, dev_y1, dev_y2, dev_y3, dev_o=shuffle_all(dev_x, dev_y1, dev_y2, dev_y3, dev_o)
	test_x, test_y1, test_y2, test_y3, test_o=shuffle_all(test_x, test_y1, test_y2, test_y3, test_o)

	feat_ids=get_feats(train_x)

	train_x=convert(train_x, feat_ids)
	dev_x=convert(dev_x, feat_ids)
	test_x=convert(test_x, feat_ids)

	scaler = StandardScaler(with_mean=False)
	scaler.fit(train_x)
	train_x=scaler.transform(train_x)
	dev_x=scaler.transform(dev_x)
	test_x=scaler.transform(test_x)

	return train_x, np.array(train_y1), np.array(train_y2), np.array(train_y3), train_o, dev_x, np.array(dev_y1), np.array(dev_y2), np.array(dev_y3), dev_o, test_x, np.array(test_y1), np.array(test_y2), np.array(test_y3), test_o, feat_ids
		

def predict(model, x, orig):
	model.eval()

	with torch.no_grad():
		for x_o, orig_batch in zip(x, orig):
			y_preds1, y_preds2, y_preds3=model.forward(x_o)
			for idx, (y_pred1, y_pred2, y_pred3, orig_url) in enumerate(zip(y_preds1, y_preds2, y_preds3, orig_batch)):

				p1=y_pred1.cpu().numpy()
				p2=y_pred2.cpu().numpy()
				p3=y_pred3.cpu().numpy()

				all_p=y_pred1.cpu().numpy() + y_pred2.cpu().numpy() + y_pred2.cpu().numpy()

				print("%s\t%.3f\t%.3f\t%.3f\t%.3f\tPRED" % (orig_url, all_p, p1, p2, p3))



def eval(x, y1, y2, y3, reg1, reg2, reg3, pred_file=None):

	preds1=reg1.predict(x)
	preds2=reg2.predict(x)
	preds3=reg3.predict(x)
	
	spearman1, _= spearmanr(preds1, y1)
	print("P1:", spearman1)
	spearman2, _= spearmanr(preds2, y2)
	print("P2:", spearman2)
	spearman3, _= spearmanr(preds3, y3)
	print("P3:", spearman3)

	all_preds=[]
	all_golds=[]

	for idx in range(len(preds1)):
		p=(preds1[idx] + preds2[idx] + preds3[idx])/3
		g=(y1[idx] + y2[idx] + y3[idx])/3
		all_preds.append(p)
		all_golds.append(g)
	
	spearman_a, _= spearmanr(all_preds, all_golds)
	print("PA:", spearman_a)

	if pred_file is not None:
		with open(pred_file, "w") as out:
			for p,g in zip(all_preds, all_golds):
				out.write("%s\t%s\n" % (p,g))

	return spearman_a

class Objective:


	def __init__(self, dataFile, feature_set):

		self.train_x, self.train_y1, self.train_y2, self.train_y3, self.train_o, self.dev_x, self.dev_y1, self.dev_y2, self.dev_y3, self.dev_o, self.test_x, self.test_y1, self.test_y2, self.test_y3, self.test_o, self.vocab=read_data(dataFile, feature_set)

	def __call__(self, trial):
		

		global bestDevScore, best1, best2, best3, best_reg

		reg_strength=trial.suggest_float("c", 1e-5, 1e9, log=True)

		reg1 = Ridge(alpha=reg_strength).fit(self.train_x, self.train_y1)
		reg2 = Ridge(alpha=reg_strength).fit(self.train_x, self.train_y2)
		reg3 = Ridge(alpha=reg_strength).fit(self.train_x, self.train_y3)

		spearman_a=eval(self.dev_x, self.dev_y1, self.dev_y2, self.dev_y3, reg1, reg2, reg3)

		if spearman_a > bestDevScore:
			bestDevScore=spearman_a
			best_reg=reg_strength
			best1=reg1
			best2=reg2
			best3=reg3


		return spearman_a


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', help='{train,predict}', required=True)
	parser.add_argument('--trainFile', help='training file (jsonl)', required=True)
	parser.add_argument('--predictionFile', help='file to predict (jsonl)', required=True)
	parser.add_argument('--concreteness_file', help='concreteness_file', required=True)
	parser.add_argument('--feature_set', help='feature_set', required=True)
	parser.add_argument('--booknlp_output_folder', help='booknlp_output_folder', required=True)

	args = vars(parser.parse_args())

	mode=args["mode"]
	trainFile=args["trainFile"]
	predictionFile=args["predictionFile"]
	concreteness_file=args["concreteness_file"]
	booknlp_output_folder=args["booknlp_output_folder"]
	feature_set=args["feature_set"]

	read_concreteness(concreteness_file)			

	if mode == "train":

		study = optuna.create_study(directions=["maximize"], pruner=optuna.pruners.HyperbandPruner())
		objective=Objective(trainFile, feature_set)
		study.optimize(objective, n_trials=50)

		print("agents features:")
		print_weights(best1, objective.vocab)

		print("events features:")
		print_weights(best2, objective.vocab)
		
		print("world features:")
		print_weights(best3, objective.vocab)
		

		print("best model: %s" % best_reg)
		print("TEST EVAL")
		spearman_a=eval(objective.test_x, objective.test_y1, objective.test_y2, objective.test_y3, best1, best2, best3, pred_file=predictionFile)



