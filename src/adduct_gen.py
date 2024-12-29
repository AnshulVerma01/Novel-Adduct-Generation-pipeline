###importing the packages and modules
import os
import sys
import pickle
import random
import argparse
from time import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from tdc import Oracle
from random import shuffle

from utils import Molecule_Dataset
import csv 
import pickle




### Defining a function for entering a input file of smiles and gives generated smiles for seed molecules as a output
def mimosa(rawdata_file):

    ### To convert the smiles into mol object
    def smiles2mol(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            return None
        Chem.Kekulize(mol)
        return mol 
    
    ### To extract the information about substructures present in the ring and outside the ring of the smiles  
    def smiles2word(smiles):
        mol = smiles2mol(smiles)
        if mol is None:
            return None 
        word_lst = []
    
        cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques_smiles = []
        for clique in cliques:
            clique_smiles = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=False)
            cliques_smiles.append(clique_smiles)
        atom_not_in_rings_list = [atom.GetSymbol() for atom in mol.GetAtoms() if not atom.IsInRing()]
        return cliques_smiles + atom_not_in_rings_list 
        

    ### vocabulary and substructure file generation    
    all_vocabulary_file = "/home/shivaji/trail/data/substructure.txt"
    #rawdata_file = "/home/shivaji/trail/data/smiles.txt"
    select_vocabulary_file = "/home/shivaji/trail/data/vocabulary.txt"
    
    if not os.path.exists(all_vocabulary_file):
    	with open(rawdata_file) as fin:
    		lines = fin.readlines()[1:]
    		smiles_lst = [line.strip().strip('"') for line in lines]
    	word2cnt = defaultdict(int)
    	for smiles in tqdm(smiles_lst):
    		word_lst = smiles2word(smiles)
    		for word in word_lst:
    			word2cnt[word] += 1
    	word_cnt_lst = [(word,cnt) for word,cnt in word2cnt.items()]
    	word_cnt_lst = sorted(word_cnt_lst, key=lambda x:x[1], reverse = True)
    
    	with open(all_vocabulary_file, 'w') as fout:
    		for word, cnt in word_cnt_lst:
    			fout.write(word + '\t' + str(cnt) + '\n')
    else:
    	with open(all_vocabulary_file, 'r') as fin:
    		lines = fin.readlines()
    		word_cnt_lst = [(line.split('\t')[0], int(line.split('\t')[1])) for line in lines]
    
    
    word_cnt_lst = list(filter(lambda x:x[1]>10, word_cnt_lst))
    print(len(word_cnt_lst))
    
    with open(select_vocabulary_file, 'w') as fout:
    	for word, cnt in word_cnt_lst:
    		fout.write(word + '\t' + str(cnt) + '\n')
    
    
    from module import GCN
    from dpp import DPPModel
    from inference_utils import optimize_single_molecule_one_iterate, oracle_screening, dpp
    from chemutils import smiles2feature, is_valid
    
    smiles_database = rawdata_file   #smiles_database = "/home/shivaji/trail/data/smiles.txt"
    clean_smiles_database = "/home/shivaji/trail/data/clean.txt"
    
    ### For filtering the smiles based on the substructure present in the vocabulary file
    with open(smiles_database, 'r') as fin:
    	lines = fin.readlines()[1:]
    smiles_lst = [i.strip().strip('"') for i in lines]
    
    clean_smiles_lst = []
    for smiles in tqdm(smiles_lst):
    	if is_valid(smiles):
    		clean_smiles_lst.append(smiles)
    clean_smiles_set = set(clean_smiles_lst)
    with open(clean_smiles_database, 'w') as fout:
    	for smiles in clean_smiles_set:
    		fout.write(smiles + '\n')
      
    
    ### For training the gnn model
    device = 'cpu'
    data_file = "/home/shivaji/trail/data/clean.txt"
    with open(data_file, 'r') as fin:
    	lines = fin.readlines()
    
    shuffle(lines)
    lines = [line.strip() for line in lines]
    N = int(len(lines) * 0.9)
    train_data = lines[:N]
    valid_data = lines[N:]
    
    
    
    training_set = Molecule_Dataset(train_data)
    valid_set = Molecule_Dataset(valid_data)
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 1}
    # exit() 
    
    
    def collate_fn(batch_lst):
    	return batch_lst
    
    train_generator = torch.utils.data.DataLoader(training_set, collate_fn = collate_fn, **params)
    valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn = collate_fn, **params)
    
    gnn = GCN(nfeat = 50, nhid = 100, num_layer = 3).to(device)
    print('GNN is built!')
    # exit() 
    
    err_t = []
    err_v = []
    cost_lst = []
    valid_loss_lst = []
    epoch = 5 
    every_k_iters = 5000
    save_folder = "/home/shivaji/trail/save_model/GNN_epoch_" 
    for ep in tqdm(range(epoch)):
    	for i, smiles in tqdm(enumerate(train_generator)):
    		### 1. training
    		smiles = smiles[0]
    		try:
    			node_mat, adjacency_matrix, idx, label = smiles2feature(smiles)  ### smiles2feature: only mask leaf node    
    		except:
    			err_t.append(smiles)
    			continue
    		# idx_lst, node_mat, substructure_lst, atomidx_2substridx, adjacency_matrix, leaf_extend_idx_pair = smiles2graph(smiles)
    		node_mat = torch.FloatTensor(node_mat).to(device)
    		adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
    		label = torch.LongTensor([label]).view(-1).to(device)
    		# print('label', label)
    		cost = gnn.learn(node_mat, adjacency_matrix, idx, label)
    		cost_lst.append(cost)
    
    		#### 2. validation 
    		if i % every_k_iters == 0:
    			gnn.eval()
    			valid_loss, valid_num = 0,0 
    			for smiles in valid_generator:
    				smiles = smiles[0]
    				try:
    					node_mat, adjacency_matrix, idx, label = smiles2feature(smiles)  
    				except:
    					err_v.append(smiles)
    					continue
    				node_mat = torch.FloatTensor(node_mat).to(device)
    				adjacency_matrix = torch.FloatTensor(adjacency_matrix).to(device)
    				label = torch.LongTensor([label]).view(-1).to(device)
    				cost, _ = gnn.infer(node_mat, adjacency_matrix, idx, label)
    				valid_loss += cost
    				valid_num += 1 
    			valid_loss = valid_loss / valid_num
    			valid_loss_lst.append(valid_loss)
    			file_name = save_folder + str(ep) + "_validloss_" + str(valid_loss)[:7] + ".ckpt"
    			torch.save(gnn, file_name)
    			gnn.train()
    
    
    
    # Open file for writing error SMILES in training data
    with open('/home/shivaji/trail/data/error_smiles_training.txt', 'w') as f:
        for item in err_t:
            f.write("%s\n" % item)
    
    # Open file for writing error SMILES in validation data
    with open('/home/shivaji/trail/data/error_smiles_validation.txt', 'w') as f:
        for item in err_v:
            f.write("%s\n" % item)
    
    
    
    
    
    #start_smiles_lst = ['N2C=NC3=C(N=C(N=C32)N)N', 'CC1=CN(C(=O)NC1=O)', 'NC1=NC2=C(N=CN2)C(=O)N1', 'N2C=CC(=NC2=O)N'] #without sugar
    #start_smiles_lst = ['C1C(C(OC1N2C=NC3=C(N=C(N=C32)N)N)CO)O', 'CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)O', 'NC1=NC2=C(N=CN2[C@H]2C[C@H](O)[C@@H](CO)O2)C(=O)N1', 'C1C(C(OC1N2C=CC(=NC2=O)N)CO)O'] #with sugar
    
    
    
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
    start_smiles_lst = ['C1C(C(OC1N2C=NC3=C(N=C(N=C32)N)N)CO)O', 'CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)O', 'NC1=NC2=C(N=CN2[C@H]2C[C@H](O)[C@@H](CO)O2)C(=O)N1', 'C1C(C(OC1N2C=CC(=NC2=O)N)CO)O'] #with sugar
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
     
         
    
    
    # Path to the directory containing the input files
    input_dir = '/home/shivaji/trail/result/pkl/'
    
    # Path to the directory where the output files will be saved
    output_dir = '/home/shivaji/trail/result/csv/'
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.startswith('smiles_') and filename.endswith('.pkl'):
            # Load the data from the input file
            with open(os.path.join(input_dir, filename), 'rb') as file:
                object_file = pickle.load(file)
    
            # Convert the data into a Pandas DataFrame
            f = pd.DataFrame(object_file[0].items())
    
            # Extract the dictionaries from the DataFrame
            dicts = [f[1][i][0] for i in range(len(f))]
    
            # Combine the dictionaries
            combined_dict = {}
            for d in dicts:
                combined_dict.update(d)
    
            # Write the contents of the dictionary (smiles and scores) to a CSV file with headings
            output_filename = os.path.splitext(filename)[0] + '.csv'
            with open(os.path.join(output_dir, output_filename), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['smiles', 'scores'])  # Write the headings to the first row of the file
                # Skip the first row (header) when writing data
                first_row = True
                for key, value in combined_dict.items():
                    if first_row:
                        first_row = False
                    else:
                        writer.writerow([key, value])
    
    

    # Directory containing the CSV files
    directory = '/home/shivaji/trail/result/csv/'  # Replace with the actual directory path
    
    # Initialize an empty DataFrame to store the merged data
    merged_data = pd.DataFrame()
    
    # List of nucleotides corresponding to each CSV file
    nucleotides = ['A', 'T', 'G', 'C']
    
    # Iterate over the CSV files and merge them
    for index, nucleotide in enumerate(nucleotides):
        # Construct the file name
        file_name = f'smiles_{index}.csv'
        file_path = os.path.join(directory, file_name)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add a 'Nucleotide' column with the corresponding nucleotide
        df['Nucleotide'] = nucleotide
        
        # Append the data to the merged DataFrame
        merged_data = pd.concat([merged_data, df], ignore_index=True)   #merged_data = merged_data.append(df, ignore_index=True)
    
    # Save the merged data to a new CSV file
    output_file = '/home/shivaji/trail/result/csv/ATGC.csv'  # Replace with the desired output file path
    merged_data.to_csv(output_file, index=False)


    
    # Load the CSV file into a DataFrame
    csv_file = '/home/shivaji/trail/result/csv/ATGC.csv'  # Replace with the path to your CSV file
    df = pd.read_csv(csv_file)
    
    # Get the list of unique nucleotides
    nucleotides = df['Nucleotide'].unique()
    
    # Iterate through nucleotides and create separate text files
    for nucleotide in nucleotides:
        # Filter the DataFrame for the current nucleotide
        nucleotide_data = df[df['Nucleotide'] == nucleotide]
    
        # Extract the SMILES data for the current nucleotide
        smiles_data = nucleotide_data['smiles']
    
        # Write the SMILES data to a text file named after the nucleotide
        output_file = f'/home/shivaji/trail/src/{nucleotide}.txt'  # Replace with the desired output directory
        with open(output_file, 'w') as file:
            for index, smiles in enumerate(smiles_data, start=1):
                file.write(f'gen_{index} {smiles}\n')
    
        # Extract only the SMILES data and store it in a different file at a different path
        smiles_output_file = f'/home/shivaji/trail/result/csv/txt/tanimoto_txt/{nucleotide}_smiles.txt'  # Replace with the desired output path
        with open(smiles_output_file, 'w') as smiles_file:
            for smiles in smiles_data:
                smiles_file.write(f'{smiles}\n')
    
    
    # # Load the CSV file into a DataFrame
    # csv_file = '/home/shivaji/trail/result/csv/ATGC.csv'  # Replace with the path to your CSV file
    # df = pd.read_csv(csv_file)
    
    # # Get the list of unique nucleotides
    # nucleotides = df['Nucleotide'].unique()
    
    # # Iterate through nucleotides and create separate text files
    # for nucleotide in nucleotides:
    #     # Filter the DataFrame for the current nucleotide
    #     nucleotide_data = df[df['Nucleotide'] == nucleotide]
    
    #     # Extract the SMILES data for the current nucleotide
    #     smiles_data = nucleotide_data['smiles']
    
    #     # Write the SMILES data to a text file named after the nucleotide
    #     output_file = f'/home/shivaji/trail/result/csv/txt/{nucleotide}.txt'  # Replace with the desired output directory
    #     with open(output_file, 'w') as file:
    #         for index, smiles in enumerate(smiles_data, start=1):
    #             file.write(f'gen_{index} {smiles}\n')
    



#mimosa('/home/shivaji/trail/data/smiles.txt')
