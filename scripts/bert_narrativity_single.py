from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoTokenizer, DebertaV2Model
from transformers import ModernBertForMaskedLM
import datetime
from collections import Counter

import optuna
import nltk, copy

import torch
import torch.nn as nn
import numpy as np
import random
from scipy.stats import spearmanr
import sys, json, argparse
from math import log
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bestOverallDev=0

def read_predict_data(filename):
	o=[]
	x=[]

	with open(filename) as file:
		for line in file:

			data=json.loads(line.rstrip())
	
			url=data["url"]

			text=data["text"]
			x.append(text)				
			o.append(url)

	return x,  o

def read_data(filename):
	o=[]
	x=[]
	y1=[]
	y2=[]
	y3=[]
	splits=[]

	with open(filename) as file:
		for line in file:

			data=json.loads(line.rstrip())

			split=data["split"]
			url=data["url"]

			agents=data["agent"]
			events=data["events"]
			world=data["world"]

			text=data["text"]
			x.append(text)
			y1.append(agents)
			y2.append(events)
			y3.append(world)
			o.append(url)
			splits.append(split)


	# shuffle the data
	tmp = list(zip(x, y1, y2, y3, o, splits))
	random.shuffle(tmp)
	x, y1, y2, y3, o, splits = zip(*tmp)
	
	return x, y1, y2, y3, o, splits
	

def predict(agent_model, event_model, world_model, x, orig, outfile):

	with torch.no_grad():
		with open(outfile, "w") as out:

			for x_o, orig_batch in zip(test_batch_x, test_batch_orig):

				with torch.no_grad():

					y_preds1=agent_model.forward(x_o)
					y_preds2=event_model.forward(x_o)
					y_preds3=world_model.forward(x_o)
									
					for idx, (y_pred1, y_pred2, y_pred3, orig_url) in enumerate(zip(y_preds1, y_preds2, y_preds3, orig_batch)):

						y_pred1=y_pred1.cpu().item()
						y_pred2=y_pred2.cpu().item()
						y_pred3=y_pred3.cpu().item()

						all_p=(y_pred1+y_pred2+y_pred3)/3

						out.write("%s\t%.3f\t%.3f\t%.3f\t%.3f\n" % (orig_url, all_p, y_pred1, y_pred2, y_pred3))


def evaluate(x, y1, y2, y3, agent_model=None, event_model=None, world_model=None, pred_file=None):

	if agent_model is not None:
		agent_model.eval()
	if event_model is not None:
		event_model.eval()
	if world_model is not None:
		world_model.eval()				

	agent_preds=[]
	agent_golds=[]

	event_preds=[]
	event_golds=[]

	world_preds=[]
	world_golds=[]


	with torch.no_grad():
		for x_o, y1_o, y2_o, y3_o in zip(x, y1, y2, y3):

			if agent_model is not None:

				y_preds1=agent_model.forward(x_o)

				for idx, y_pred1 in enumerate(y_preds1):
					agent_golds.append(y1_o[idx].cpu().numpy())
					agent_preds.append(y_pred1.cpu().numpy())

			if event_model is not None:

				y_preds1=event_model.forward(x_o)

				for idx, y_pred1 in enumerate(y_preds1):
					event_golds.append(y2_o[idx].cpu().numpy())
					event_preds.append(y_pred1.cpu().numpy())

			if world_model is not None:

				y_preds1=world_model.forward(x_o)

				for idx, y_pred1 in enumerate(y_preds1):
					world_golds.append(y3_o[idx].cpu().numpy())
					world_preds.append(y_pred1.cpu().numpy())


	if pred_file is not None:
		with open(pred_file, "w") as out:
			for ap,ep,wp,ag,eg,wg in zip(agent_preds, event_preds, world_preds, agent_golds, event_golds, world_golds):
				big_gold=(ag+eg+wg)/3
				big_pred=(ap+ep+wp)/3
				
				out.write("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (big_pred, big_gold, ap,ep,wp,ag,eg,wg))

	if agent_model is not None:
		spearman1, _= spearmanr(agent_preds, agent_golds)
		print("P1:", spearman1)
		return spearman1
	if event_model is not None:
		spearman2, _= spearmanr(event_preds, event_golds)
		print("P2:", spearman2)
		return spearman2

	if world_model is not None:
		spearman3, _= spearmanr(world_preds, world_golds)
		print("P3:", spearman3)
		return spearman3
	


class BERTRegressor(nn.Module):

	
	def __init__(self, params):
		super().__init__()

		if base == "deberta":
				# self.model_name="deberta"
				self.model_name="deberta-base"
				self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
				self.bert=model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")

		elif base == "roberta":

			self.model_name="roberta-base"
			self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False)
			self.bert = RobertaModel.from_pretrained(self.model_name)


		elif base == "modern-bert":

			self.model_name="modern-bert"

			self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", do_lower_case=False, do_basic_tokenize=False)
			self.bert = ModernBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")


		else:
			
			self.model_name="bert-base-cased"
			self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False)
			self.bert = BertModel.from_pretrained(self.model_name)


		print("Model:", self.model_name)

		self.fc = nn.Linear(768, 1)
		

	def get_predict_batches(self, all_x, all_orig, batch_size=12, max_toks=512):

		batches_x=[]
		batches_orig=[]
		
		for i in range(0, len(all_x), batch_size):

			current_batch=[]

			x=all_x[i:i+batch_size]

			o=all_orig[i:i+batch_size]
			
			batch_x = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt", max_length=max_toks)
			
			batches_x.append(batch_x)
			batches_orig.append(o)

		return batches_x, batches_orig
  

	def get_batches(self, all_x, all_y1, all_y2, all_y3, all_orig, batch_size=12, max_toks=512):

		batches_x=[]
		batches_y1=[]
		batches_y2=[]
		batches_y3=[]
		batches_orig=[]
		
		for i in range(0, len(all_x), batch_size):

			current_batch=[]

			x=all_x[i:i+batch_size]

			o=all_orig[i:i+batch_size]
		
			batch_x = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt", max_length=max_toks)

			batch_y1=all_y1[i:i+batch_size]
			batch_y2=all_y2[i:i+batch_size]
			batch_y3=all_y3[i:i+batch_size]

			batches_x.append(batch_x)
			batches_y1.append(torch.FloatTensor(batch_y1))
			batches_y2.append(torch.FloatTensor(batch_y2))
			batches_y3.append(torch.FloatTensor(batch_y3))
			batches_orig.append(o)

		return batches_x, batches_y1, batches_y2, batches_y3, batches_orig
  
	def forward(self, batch_x): 
		
		bert_output1 = self.bert(input_ids=batch_x["input_ids"].to(device),
				 attention_mask=batch_x["attention_mask"].to(device),
				 output_hidden_states=True)
		bert_hidden_states1 = bert_output1['hidden_states']
		out1 = bert_hidden_states1[-1][:,0,:]
		out1 = self.fc(out1).squeeze()

		return out1


	def captum_forward(self, input_ids, attention_mask, task):
		print("ids", input_ids.shape)
		print("mask", attention_mask.shape)
		
		bert_output1 = self.bert1(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
		bert_hidden_states1 = bert_output1['hidden_states']
		out1 = bert_hidden_states1[-1][:,0,:]
		out1 = self.fc1(out1)

		bert_output2 = self.bert2(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
		bert_hidden_states2 = bert_output2['hidden_states']
		out2 = bert_hidden_states2[-1][:,0,:]
		out2 = self.fc1(out2)

		bert_output3 = self.bert3(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
		bert_hidden_states3 = bert_output3['hidden_states']
		out3 = bert_hidden_states3[-1][:,0,:]
		out3 = self.fc1(out3)		


		if task == 1:
			return out1.squeeze(-1)
		elif task == 2:
			return out2.squeeze(-1)
		elif task == 3:
			return out3.squeeze(-1)





def split_data(split, splits, all_x, all_y1, all_y2, all_y3, orig):
	new_x=[]
	new_y1=[]
	new_y2=[]
	new_y3=[]
	new_orig=[]

	for idx, url in enumerate(orig):
		
		if splits[idx] == split:
			new_x.append(all_x[idx])
			new_y1.append(all_y1[idx])
			new_y2.append(all_y2[idx])
			new_y3.append(all_y3[idx])
			new_orig.append(orig[idx])

	return new_x, new_y1, new_y2, new_y3, new_orig
			

class Objective:

	def __init__(self, modelName, base, trainFile, device_map, device, task):

		self.modelName=modelName
		self.base=base
		self.trainFile=trainFile
		self.task=task

		self.device_map=device_map
		self.device=device


	def __call__(self, trial):
		
		learningRate=trial.suggest_float("lr", 1e-6, 5e-3, log=True)

		global bestOverallDev

		start_time = datetime.datetime.now()

		all_x, all_y1, all_y2, all_y3, orig, splits=read_data(self.trainFile)
		
		train_x, train_y1, train_y2, train_y3, train_orig=split_data("train", splits, all_x, all_y1, all_y2, all_y3, orig)
		dev_x, dev_y1, dev_y2, dev_y3, dev_orig=split_data("dev", splits, all_x, all_y1, all_y2, all_y3, orig)
		test_x, test_y1, test_y2, test_y3, test_orig=split_data("test", splits, all_x, all_y1, all_y2, all_y3, orig)

		print("Train n: %s, dev n: %s, test n: %s, task: %s" % (len(train_orig), len(dev_orig), len(test_orig), self.task))

		for url in test_orig:
			assert (url not in train_orig and url not in dev_orig)
		for url in dev_orig:
			assert url not in train_orig

		best_dev_acc = 0.

		bert_model = BERTRegressor(params={"base": base})
		bert_model.to(device)

		batch_x, batch_y1, batch_y2, batch_y3, batch_orig = bert_model.get_batches(train_x, train_y1, train_y2, train_y3, train_orig)
		dev_batch_x, dev_batch_y1, dev_batch_y2, dev_batch_y3, dev_batch_orig = bert_model.get_batches(dev_x, dev_y1, dev_y2, dev_y3, dev_orig)
		test_batch_x, test_batch_y1, test_batch_y2, test_batch_y3, test_batch_orig = bert_model.get_batches(test_x, test_y1, test_y2, test_y3, test_orig)

		optimizer = torch.optim.Adam(bert_model.parameters(), lr=learningRate)

		loss_fn=nn.MSELoss()

		num_epochs=100

		best_epoch=0
		patience=10

		for epoch in range(num_epochs):
			bert_model.train()

			bigloss=0
			for x, y1, y2, y3 in zip(batch_x, batch_y1, batch_y2, batch_y3):

				y=None

				if task == 1:
					y=y1
				elif task == 2:
					y=y2
				elif task == 3:
					y=y3

				y_pred1 = bert_model.forward(x)

				loss = loss_fn(y_pred1.view(-1, 1), y.to(device).view(-1,1)) 

				bigloss+=loss
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			print("loss", bigloss)

			if task == 1:
				dev_accuracy=evaluate(dev_batch_x, dev_batch_y1, dev_batch_y2, dev_batch_y3, agent_model=bert_model)
			elif task == 2:
				dev_accuracy=evaluate(dev_batch_x, dev_batch_y1, dev_batch_y2, dev_batch_y3, event_model=bert_model)
			elif task == 3:
				dev_accuracy=evaluate(dev_batch_x, dev_batch_y1, dev_batch_y2, dev_batch_y3, world_model=bert_model)

			print("INTERMEDIATE\t%s\t%.3f\t%s\t%s" % (epoch, dev_accuracy, learningRate, trial))
			sys.stdout.flush()

			if dev_accuracy > best_dev_acc:
				best_dev_acc = dev_accuracy
				best_epoch=epoch

			if dev_accuracy > bestOverallDev:
				torch.save(bert_model.state_dict(), self.modelName)
				bestOverallDev=dev_accuracy

			if epoch-best_epoch > patience:
				print("No change in %s epochs, exiting" % patience)
				break

			trial.report(dev_accuracy, epoch)
			if trial.should_prune():
				raise optuna.TrialPruned()

		end_time = datetime.datetime.now()
		elapsed_time = end_time - start_time

		print("LRperf\t%s\t%.3f\t%.1f" % (learningRate, best_dev_acc, elapsed_time.total_seconds()/60))
		sys.stdout.flush()

		return best_dev_acc


def viz_all(base, agent_model_file, event_model_file, world_model_file, filename):

	agent_model = BERTRegressor(params={"base": base})
	agent_model.load_state_dict(torch.load(agent_model_file))
	agent_model.eval()
	agent_model.to(device)

	event_model = BERTRegressor(params={"base": base})
	event_model.load_state_dict(torch.load(event_model_file))
	event_model.eval()
	event_model.to(device)

	world_model = BERTRegressor(params={"base": base})
	world_model.load_state_dict(torch.load(world_model_file))
	world_model.eval()
	world_model.to(device)

	agents_all=Counter()
	events_all=Counter()
	world_all=Counter()
	with open(filename) as file:
		for idx, line in enumerate(file):
			data=json.loads(line.rstrip())
			text=data["text"]

			agents, events, world=interpret_one(text, agent_model, event_model, world_model)
			
			for word in set(agents):
				agents_all[word.lower()]+=1
			for word in set(events):
				events_all[word.lower()]+=1
			for word in set(world):
				world_all[word.lower()]+=1	


			for k,v in agents_all.most_common(25):
				print("%s AGENT\t%s\t%s" % (idx, k,v))	
			print()				

			for k,v in events_all.most_common(25):
				print("%s EVENT\t%s\t%s" % (idx, k,v))	
			print()

			for k,v in world_all.most_common(25):
				print("%s WORLD\t%s\t%s" % (idx, k,v))	
			print()
			print("===============================")
			print()
			sys.stdout.flush()


def interpret_one(text, agent_model, event_model, world_model):

	tokens=nltk.word_tokenize(text)

	test_x=[]
	test_orig=[]

	test_x.append(' '.join(tokens))
	test_orig.append("__ORIGINAL__")

	for idx, tok in enumerate(tokens):
		cop=copy.deepcopy(tokens)
		cop[idx]=agent_model.tokenizer.mask_token
		test_x.append(' '.join(cop))
		test_orig.append(tok)

	test_batch_x, test_batch_orig = agent_model.get_predict_batches(test_x, test_orig)

	base_pred1=base_pred2=base_pred3=None
	all_preds=[]

	for x_o, orig_batch in zip(test_batch_x, test_batch_orig):

		with torch.no_grad():

			y_preds1=agent_model.forward(x_o)
			y_preds2=event_model.forward(x_o)
			y_preds3=world_model.forward(x_o)
				
			if y_preds1.dim() == 0:
				y_preds1 = y_preds1.unsqueeze(0)
			if y_preds2.dim() == 0:
				y_preds2 = y_preds2.unsqueeze(0)
			if y_preds3.dim() == 0:
				y_preds3 = y_preds2.unsqueeze(0)
				
			for idx, (y_pred1, y_pred2, y_pred3, orig_token) in enumerate(zip(y_preds1, y_preds2, y_preds3, orig_batch)):

				y_pred1=y_pred1.cpu().item()
				y_pred2=y_pred2.cpu().item()
				y_pred3=y_pred3.cpu().item()

				if orig_token == "__ORIGINAL__":
					base_pred1=y_pred1
					base_pred2=y_pred2
					base_pred3=y_pred3
				else:
					all_preds.append((y_pred1, y_pred2, y_pred3, orig_token))
	
	diffs1=[]
	diffs2=[]
	diffs3=[]
	
	for y_pred1, y_pred2, y_pred3, orig_token in all_preds:
		diff1=y_pred1
		diff2=y_pred2
		diff3=y_pred3

		diffs1.append((diff1, orig_token))
		diffs2.append((diff2, orig_token))
		diffs3.append((diff3, orig_token))

	def printer(diffs):
		for k,v in sorted(diffs)[:10]:
			print(k,v)
		print()

	def get_top(diffs):
		print("DATA\t%s" % (json.dumps(diffs)))
		diffs=sorted(diffs)
		listt=[]
		for k,v in diffs[:10]:
			listt.append(v.lower())
		return listt


	printer(diffs1)
	printer(diffs2)
	printer(diffs3)

	return get_top(diffs1), get_top(diffs2), get_top(diffs3)
						


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', help='{train,predict}', required=True)
	parser.add_argument('--trainFile', help='training file (jsonl)', required=False)
	parser.add_argument('--predictionFile', help='file to predict (jsonl)', required=False)
	# parser.add_argument('--modelName', help='model name', required=True)
	parser.add_argument('--base', help='bert,deberta,roberta', required=True)
	parser.add_argument('--device', help='auto or gpu number', required=False)
	parser.add_argument('--task', help='{agents, events, world}', required=False)
	parser.add_argument('--agent_model', help='agent model filename', required=False)
	parser.add_argument('--event_model', help='event model filename', required=False)
	parser.add_argument('--world_model', help='world model filename', required=False)

	args = vars(parser.parse_args())
	mode=args["mode"]
	trainFile=args["trainFile"]
	predictionFile=args["predictionFile"]
	base=args["base"]
	deviceName=args["device"]
	task=args["task"]
	
	agent_model=args["agent_model"]
	event_model=args["event_model"]
	world_model=args["world_model"]

	if task == "agents":
		task=1
		modelName=agent_model
	elif task == "events":
		task=2
		modelName=event_model
	elif task == "world":
		task=3
		modelName=world_model

	if deviceName is None or deviceName == "auto":
		device_map="auto"
		device="cuda"

	else:
		device = torch.device(deviceName if torch.cuda.is_available() else "cpu")
		device_map=device

	if mode == "train":

		study = optuna.create_study(directions=["maximize"], pruner=optuna.pruners.HyperbandPruner())

		study.optimize(Objective(modelName, base, trainFile, device_map, device, task), n_trials=50)

	elif mode == "evaluate":

		all_x, all_y1, all_y2, all_y3, orig, splits=read_data(trainFile)
		test_x, test_y1, test_y2, test_y3, test_orig=split_data("test", splits, all_x, all_y1, all_y2, all_y3, orig)
		
		bert_model_agent = BERTRegressor(params={"base": base})
		bert_model_agent.load_state_dict(torch.load(agent_model))
		bert_model_agent.to(device)

		bert_model_events = BERTRegressor(params={"base": base})
		bert_model_events.load_state_dict(torch.load(event_model))
		bert_model_events.to(device)

		bert_model_world = BERTRegressor(params={"base": base})
		bert_model_world.load_state_dict(torch.load(world_model))
		bert_model_world.to(device)

		test_batch_x, test_batch_y1, test_batch_y2, test_batch_y3, test_batch_orig = bert_model_agent.get_batches(test_x, test_y1, test_y2, test_y3, test_orig)

		test_accuracy=evaluate(test_batch_x, test_batch_y1, test_batch_y2, test_batch_y3, agent_model=bert_model_agent, event_model=bert_model_events, world_model=bert_model_world, pred_file=predictionFile)



	elif mode == "predict":

		test_x, test_orig=read_predict_data(trainFile)

		agent_model_o = BERTRegressor(params={"base": base})
		agent_model_o.load_state_dict(torch.load(agent_model))
		agent_model_o.eval()
		agent_model_o.to(device)

		event_model_o = BERTRegressor(params={"base": base})
		event_model_o.load_state_dict(torch.load(event_model))
		event_model_o.eval()
		event_model_o.to(device)

		world_model_o = BERTRegressor(params={"base": base})
		world_model_o.load_state_dict(torch.load(world_model))
		world_model_o.eval()
		world_model_o.to(device)


		test_batch_x, test_batch_orig = agent_model_o.get_predict_batches(test_x, test_orig)
		predict(agent_model_o, event_model_o, world_model_o, test_batch_x, test_batch_orig, predictionFile)
	
	elif mode == "viz_all":

		viz_all(base, agent_model, event_model, world_model, trainFile)

