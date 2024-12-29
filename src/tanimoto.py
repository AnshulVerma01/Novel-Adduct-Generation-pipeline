import os
import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import seaborn as sns
import matplotlib.pyplot as plt
############## exp and put files should be in same directory


def tanimoto_score(input_dir, exp_dir):
    
    input_files = ['A_smiles.txt', 'T_smiles.txt','G_smiles.txt','C_smiles.txt']
    exp_files = ['exp_a.txt', 'exp_t.txt','exp_g.txt','exp_c.txt']

    for input_file, exp_file in zip(input_files, exp_files):
        # Load the SMILES from the input file into a list
        with open(input_file, 'r') as file:
            smiles_A = [line.strip() for line in file]
        # Load the SMILES from the experimental file into a list
        with open(exp_file, 'r') as file:
            smiles_exp = [line.strip() for line in file]
        # Create lists to store SMILES_exp, top similarity values, and corresponding SMILES_A
        exp_data = []
        # Convert SMILES to fingerprints for the input file
        mols_A = [Chem.MolFromSmiles(smiles) for smiles in smiles_A]
        fps_A = [Chem.RDKFingerprint(mol) for mol in mols_A]
        for smiles_exp in smiles_exp:
            mol_exp = Chem.MolFromSmiles(smiles_exp)
            if mol_exp is not None:
                fp_exp = Chem.RDKFingerprint(mol_exp)
                similarities = [DataStructs.TanimotoSimilarity(fp_exp, fp) for fp in fps_A]
                max_similarity = max(similarities)
                max_similarity_index = similarities.index(max_similarity)
                exp_data.append([smiles_exp, max_similarity, smiles_A[max_similarity_index]])
        # Save the results to a CSV file
        output_file = input_file.replace('.txt', '_similarities.csv')
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Smiles', 'Top_Similarity', {input_file}])
            for data in exp_data:
                writer.writerow(data)
        # Plot a frequency histogram using Seaborn
        plt.figure()
        similarity_scores = [data[1] for data in exp_data]
        sns.histplot(similarity_scores, bins=20, kde=True, color='blue')
        plt.xlabel('Tanimoto Similarity')
        plt.ylabel('Frequency')
        plt.title(f'Frequency Distribution of the Top Tanimoto Similarity for {exp_file.replace("_exp.txt", "")} vs adducts')
        plt.savefig(f'{input_file}_vs_{exp_file}_histogram.png')

    exp_files= ['put_a.txt' ,'put_t.txt','put_g.txt','put_c.txt']

    for input_file, exp_file in zip(input_files, exp_files):
        # Load the SMILES from the input file into a list
        with open(input_file, 'r') as file:
            smiles_A = [line.strip() for line in file]
        # Load the SMILES from the experimental file into a list
        with open(exp_file, 'r') as file:
            smiles_exp = [line.strip() for line in file]
        # Create lists to store SMILES_exp, top similarity values, and corresponding SMILES_A
        exp_data = []
        # Convert SMILES to fingerprints for the input file
        mols_A = [Chem.MolFromSmiles(smiles) for smiles in smiles_A]
        fps_A = [Chem.RDKFingerprint(mol) for mol in mols_A]
        for smiles_exp in smiles_exp:
            mol_exp = Chem.MolFromSmiles(smiles_exp)
            if mol_exp is not None:
                fp_exp = Chem.RDKFingerprint(mol_exp)
                similarities = [DataStructs.TanimotoSimilarity(fp_exp, fp) for fp in fps_A]
                max_similarity = max(similarities)
                max_similarity_index = similarities.index(max_similarity)
                exp_data.append([smiles_exp, max_similarity, smiles_A[max_similarity_index]])
        # Save the results to a CSV file
        output_file = input_file.replace('.txt', '_similarities.csv')
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Smiles', 'Top_Similarity', {input_file}])
            for data in exp_data:
                writer.writerow(data)
        # Plot a frequency histogram using Seaborn
        plt.figure()
        similarity_scores = [data[1] for data in exp_data]
        sns.histplot(similarity_scores, bins=20, kde=True, color='blue')
        plt.xlabel('Tanimoto Similarity')
        plt.ylabel('Frequency')
        plt.title(f'Frequency Distribution of the Top Tanimoto Similarity for {exp_file.replace("_exp.txt", "")} vs adducts')
        plt.savefig(f'{input_file}_vs_{exp_file}_histogram.png')

# Specify the directories containing input and experimental files directly within the function
# input_dir = '/home/shivaji/trail/result/csv/txt/tanimoto_txt'
# exp_dir = '/home/shivaji/trail/result/csv/txt/tanimoto_txt/adducts'

# Call the function to process files in the specified directories
# tanimoto_score(input_dir, exp_dir)


def tanimoto_heatmap(input_dir, adduct_dir):
    # List of input and adduct filenames
    input_files = ['A_smiles.txt', 'T_smiles.txt','G_smiles.txt','C_smiles.txt']
    adduct_files = ['exp_a.txt', 'exp_t.txt','exp_g.txt','exp_c.txt']

    for input_file, adduct_file in zip(input_files, adduct_files):
        # Read SMILES data from the input and adduct files
        with open(input_file, 'r') as file:
            gen_adducts = [line.strip() for line in file]

        with open(adduct_file, 'r') as file:
            adducts = [line.strip() for line in file]

        smiles_list = []
        similarity_matrix = []

        # Generate RDKit fingerprints for input SMILES
        fps_A = [Chem.RDKFingerprint(Chem.MolFromSmiles(smiles)) for smiles in gen_adducts]

        # Calculate Tanimoto similarity for each adduct
        for smiles_exp in adducts:
            mol_exp = Chem.MolFromSmiles(smiles_exp)
            if mol_exp is not None:
                fp_exp = Chem.RDKFingerprint(mol_exp)
                similarities = [DataStructs.TanimotoSimilarity(fp_exp, fp) for fp in fps_A]
                smiles_list.append(smiles_exp)
                similarity_matrix.append(similarities)

        # Convert the similarity matrix to a NumPy array
        similarity_matrix = np.array(similarity_matrix)

        # Create a heatmap
        plt.figure(figsize=(10, 8))
        sns.clustermap(similarity_matrix, xticklabels=False, yticklabels=False, cmap="viridis", annot=False,vmin=0, vmax=1, col_cluster=False,dendrogram_ratio=(.1, .2), cbar_pos=(0, .2, .03, .4), fmt=".4f", cbar_kws={'label': 'Tanimoto similarity'})

        plt.xlabel('SMILES' + input_file)
        plt.ylabel('SMILES' + adduct_file)
        #plt.title(' Tanimoto Similarity Heatmap')

        # Save the heatmap as an image file
        plt.savefig(f'{input_file}_vs_{adduct_file}_heatmap.png')

    adduct_files= ['put_a.txt', 'put_t.txt','put_g.txt','put_c.txt']
    
    for input_file, adduct_file in zip(input_files, adduct_files):
        # Read SMILES data from the input and adduct files
        with open(input_file, 'r') as file:
            gen_adducts = [line.strip() for line in file]

        with open(adduct_file, 'r') as file:
            adducts = [line.strip() for line in file]

        smiles_list = []
        similarity_matrix = []

        # Generate RDKit fingerprints for input SMILES
        fps_A = [Chem.RDKFingerprint(Chem.MolFromSmiles(smiles)) for smiles in gen_adducts]

        # Calculate Tanimoto similarity for each adduct
        for smiles_exp in adducts:
            mol_exp = Chem.MolFromSmiles(smiles_exp)
            if mol_exp is not None:
                fp_exp = Chem.RDKFingerprint(mol_exp)
                similarities = [DataStructs.TanimotoSimilarity(fp_exp, fp) for fp in fps_A]
                smiles_list.append(smiles_exp)
                similarity_matrix.append(similarities)

        # Convert the similarity matrix to a NumPy array
        similarity_matrix = np.array(similarity_matrix)

        # Create a heatmap
       # Adjust the figsize as needed

        sns.clustermap(similarity_matrix, xticklabels=False, yticklabels=False, cmap="viridis", annot=False,vmin=0, vmax=1, col_cluster=False,dendrogram_ratio=(.1, .1), cbar_pos=(0, .2, .03, .4), fmt=".4f", cbar_kws={'label': 'Tanimoto similarity'})

# Add a color bar at the bottom
        plt.xlabel('SMILES in ' + input_file)
        plt.ylabel('SMILES in ' + adduct_file)
        #plt.title('Tanimoto Similarity Heatmap')

        # Save the heatmap as an image file
        plt.savefig(f'{input_file}_vs_{adduct_file}_heatmap.png')
  
# Specify the directories containing input and adduct files directly within the function
# input_dir = '/home/shivaji/trail/result/csv/txt/tanimoto_txt'
# adduct_dir = '/home/shivaji/trail/result/csv/txt/tanimoto_txt/adducts'

# Call the function to process files in the specified directories
# tanimoto_heatmap(input_dir, adduct_dir)
