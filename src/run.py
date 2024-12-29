import os, pickle, torch, random
import numpy as np 
import argparse
from time import time
from tqdm import tqdm 
from matplotlib import pyplot as plt
from random import shuffle 
import torch.nn as nn
import torch.nn.functional as F
from tdc import Oracle
torch.manual_seed(1)
np.random.seed(2)
random.seed(1)
from chemutils import * 
from inference_utils import * 

#start_smiles_lst = ['N2C=NC3=C(N=C(N=C32)N)N', 'CC1=CN(C(=O)NC1=O)', 'NC1=NC2=C(N=CN2)C(=O)N1', 'N2C=CC(=NC2=O)N'] #without sugar
start_smiles_lst = ['C1C(C(OC1N2C=NC3=C(N=C(N=C32)N)N)CO)O', 'CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)O', 'NC1=NC2=C(N=CN2[C@H]2C[C@H](O)[C@@H](CO)O2)C(=O)N1', 'C1C(C(OC1N2C=CC(=NC2=O)N)CO)O'] #with sugar



def optimization(start_smiles_lst, gnn, oracle, oracle_num, oracle_name, generations, population_size, lamb, topk, epsilon, result_pkl):
	smiles2score = dict() ### oracle_num
	def oracle_new(smiles):
		if smiles not in smiles2score:
			value = oracle(smiles) 
			smiles2score[smiles] = value 
		return smiles2score[smiles] 
	trace_dict = dict() 
	existing_set = set(start_smiles_lst)  
	current_set = set(start_smiles_lst)
	average_f = np.mean([oracle_new(smiles) for smiles in current_set])
	f_lst = [(average_f, 0.0)]
	idx_2_smiles2f = {}
	smiles2f_new = {smiles:oracle_new(smiles) for smiles in start_smiles_lst} 
	idx_2_smiles2f[-1] = smiles2f_new, current_set 
	for i_gen in tqdm(range(generations)):
		next_set = set()
		for smiles in current_set:
			#print(smiles)
			smiles_set = optimize_single_molecule_one_iterate(smiles, gnn)
			#smiles_set = optimize_single_molecule_all_generations(smiles, gnn, oracle, generations, population_size, lamb)
			#print(smiles_set)
			#for item in smiles_set:
			#	smiles = item[0]
			#	smiles_set.append(smiles)
			

			for smi in smiles_set:
				#print(smi)
				if smi not in trace_dict:
					trace_dict[smi] = smiles ### ancestor -> offspring 
			next_set = next_set.union(smiles_set)
		# next_set = next_set.difference(existing_set)   ### if allow repeat molecule  
		smiles_score_lst = oracle_screening(next_set, oracle_new)  ###  sorted smiles_score_lst 
		print(smiles_score_lst[:5], "Oracle num", len(smiles2score))

		#current_set = [i[0] for i in smiles_score_lst[:population_size]]  # Option I: top-k 
		current_set,_,_ = dpp(smiles_score_lst = smiles_score_lst, num_return = population_size, lamb = lamb) 	# Option II: DPP
		existing_set = existing_set.union(next_set)

		# save 
		smiles2f_new = {smiles:score for smiles,score in smiles_score_lst} 
		idx_2_smiles2f[i_gen] = smiles2f_new, current_set 
		pickle.dump((idx_2_smiles2f, trace_dict), open(result_pkl, 'wb'))

		#### compute f-score
		score_lst = [smiles2f_new[smiles] for smiles in current_set] 
		average_f = np.mean(score_lst)
		std_f = np.std(score_lst)
		f_lst.append((average_f, std_f))
		str_f_lst = [str(i[0])[:5]+'\t'+str(i[1])[:5] for i in f_lst]
		with open("/home/shivaji/trail/result/" + oracle_name + "_f_t.txt", 'w') as fout:
			fout.write('\n'.join(str_f_lst))
		if len(smiles2score) > oracle_num: 
			break 


parser = argparse.ArgumentParser()
parser.add_argument('--oracle_num', type=int, default=20000)
parser.add_argument('--oracle_name', type=str, default="sa", choices=['jnkgsk', 'qedsajnkgsk', 'qed', 'jnk', 'gsk', 'logP', 'sa'])	
parser.add_argument('--generations', type=int, default=1200)	
parser.add_argument('--population_size', type=int, default=800)	
args, unknown = parser.parse_known_args() #args = parser.parse_args()
oracle_num = args.oracle_num 
oracle_name = args.oracle_name 
generations = args.generations 
population_size = args.population_size
#global start_smiles_lst

#start_smiles_lst = ['N2C=NC3=C(N=C(N=C32)N)N', 'CC1=CN(C(=O)NC1=O)', 'NC1=NC2=C(N=CN2)C(=O)N1', 'N2C=CC(=NC2=O)N'] #without sugar
#start_smiles_lst = ['C1C(C(OC1N2C=NC3=C(N=C(N=C32)N)N)CO)O', 'CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)O', 'NC1=NC2=C(N=CN2[C@H]2C[C@H](O)[C@@H](CO)O2)C(=O)N1', 'C1C(C(OC1N2C=CC(=NC2=O)N)CO)O'] #with sugar
qed = Oracle('qed')
sa = Oracle('sa')
jnk = Oracle('JNK3')
gsk = Oracle('GSK3B')
logp = Oracle('logp')
mu = 2.230044
sigma = 0.6526308
def normalize_sa(smiles):
	sa_score = sa(smiles)
	mod_score = np.maximum(sa_score, mu)
	return np.exp(-0.5 * np.power((mod_score - mu) / sigma, 2.)) 
if oracle_name == 'jnkgsk':
	def oracle(smiles):
		return np.mean((jnk(smiles), gsk(smiles)))
elif oracle_name == 'qedsajnkgsk':
	def oracle(smiles):
		return np.mean((qed(smiles), normalize_sa(smiles), jnk(smiles), gsk(smiles))) 
elif oracle_name == 'qed':
	def oracle(smiles):
		return qed(smiles) 
elif oracle_name == 'jnk':
	def oracle(smiles):
		return jnk(smiles)
elif oracle_name == 'gsk':
	def oracle(smiles):
		return gsk(smiles) 
elif oracle_name == 'logP':
	def oracle(smiles):
		return logp(smiles)
elif oracle_name == 'sa':
	def oracle(smiles):
		return sa(smiles)
device = 'cpu' ## cpu is better 

# Define the folder where the files are located
folder_path = "/home/shivaji/trail/save_model/"
# Initialize variables to hold the highest episode number and lowest validation loss
max_ep = 0
min_loss = float('inf')
best_file = ""
# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the filename matches the pattern
	if "GNN_epoch_" in filename and "_validloss_" in filename and filename.endswith(".ckpt"):
        # Extract the episode number and validation loss from the filename
		parts = filename.split("_")
		ep = int(parts[2])
		loss = float(parts[4][:7])
       	# Check if this file has a lower validation loss than the current best file
		if loss < min_loss:
			max_ep = ep
			min_loss = loss
			best_file = filename
# Print the name of the best file
print(best_file)

model_ckpt = "/home/shivaji/trail/save_model/" + best_file
 ###	device = 'cpu' ## cpu is better 
gnn = torch.load(model_ckpt)
gnn.switch_device(device)

for i, start_smiles in enumerate(start_smiles_lst):
	result_pkl = "/home/shivaji/trail/result/pkl/smiles_" + str(i)+ ".pkl"
	optimization([start_smiles], gnn, oracle, oracle_num, oracle_name,
						generations = generations, 
						population_size = population_size, 
						lamb=2, 
						topk = 5, 
						epsilon = 0.7, 
						result_pkl = result_pkl)