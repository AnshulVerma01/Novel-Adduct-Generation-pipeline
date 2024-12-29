import warnings
from tqdm import tqdm
import pandas as pd
import sys
import os
from matchms.importing import load_from_msp, load_from_mzml
from matchms.filtering import default_filters, normalize_intensities
from matchms import calculate_scores
from matchms.similarity import CosineGreedy
import csv
# To silence all warnings
warnings.filterwarnings("ignore")  
# # imports query spectra file of mzml format
# def import_spectra(input_file=str, MS_value=int):
#     file_q = list(load_from_mzml(input_file, MS_value))
#     return file_q    
    
    # #import ms spectral libraries in msp format
    # def import_library(input_file=str, MS_value=int):
    #     file_lib = list(load_from_msp(input_file, MS_value))
    #     return file_lib

def spectra_viz(input_A,input_T,input_G,input_C,adducts,put_adducts,out_adducts_csv_path,out_put_csv_path, image_path) :   
    #normalize spectra
    def spectra_preprocess(s):
        s = default_filters(s)
        s = normalize_intensities(s)
        return s
    
    #compute cosine similarities
    def match_spectra(query_spectra, reference_spectra):
        batch_size = 100
        rows = []
    
        for i in tqdm(range(0, len(query_spectra), batch_size)):
            utm_batch = query_spectra[i:i + batch_size]
            scores = calculate_scores(references=reference_spectra,
                                      queries=utm_batch,
                                      similarity_function=CosineGreedy(),
                                      is_symmetric=False)
    
            for j, query_spectrum in enumerate(utm_batch):
                best_matches = scores.scores_by_query(query_spectrum,True) #'CosineGreedy_score',)
                for (reference, score) in best_matches[:1]:#len(reference_spectra)]:
                    query_num = f"{query_spectra}{i + j}"
                    query_smile=query_spectra[i+j].metadata['smiles/inchi']
                    query_collision_energy=query_spectra[i+j].metadata['comment']
                    reference_id = reference.metadata['id']
                    #seed = reference.metadata['seed']
                    smiles_inchi = reference.metadata['smiles/inchi']
                    Collision_Energy = reference.metadata['comment']
                    score_value = f"{score[0]:.4f}"
                    num_matching_peaks = score[1]
                    rows.append([query_num, query_smile,query_collision_energy, reference_id, smiles_inchi, Collision_Energy, score_value, num_matching_peaks])
                    # if score[0] > 0.9:
                    #     print(f"Found hit: {query_num}, {library}, Reference ID={reference_id}, Smiles/InChI={smiles_inchi}, {Collision_Energy}, Score={score_value}, Number of Matching Peaks={num_matching_peaks}")
    
        df = pd.DataFrame(rows, columns=["Query",'Query_Smile','query_collision_energy', "Reference Scan ID", "ref_Smiles/Inchi", "ref_Collision_Energy", "Score", "Number of Matching Peaks"])
        return df

    
    
    ##load and save mimosa generated smiles cfmid spectra 
    Generated_adducts_a=list(load_from_msp(input_A))
    Generated_adducts_t=list(load_from_msp(input_T))
    Generated_adducts_c=list(load_from_msp(input_G))
    Generated_adducts_g=list(load_from_msp(input_C))
    
    
    ##experimental adducts library
    DNA_adducts= list(load_from_msp(adducts))
    Putative_adducts=list(load_from_msp(put_adducts))
    
    dna_adducts=[spectra_preprocess(s) for s in DNA_adducts]
    put_adducts=[spectra_preprocess(s) for s in Putative_adducts]
    seed_a=[spectra_preprocess(s) for s in Generated_adducts_a]
    
    
    
    seed_t=[spectra_preprocess(s) for s in Generated_adducts_t]
    
    seed_c=[spectra_preprocess(s) for s in Generated_adducts_c]
    
    seed_g=[spectra_preprocess(s) for s in Generated_adducts_g]
    
    
    def filter_spectra(input_list, metadata_key, metadata_value):
        new_list = []
        for element in input_list:
            if element.metadata.get(metadata_key) == metadata_value:
                new_list.append(element)
        return new_list
    
    
    
    dna_energy0=filter_spectra(dna_adducts,'comment','Energy0')
    dna_energy1=filter_spectra(dna_adducts,'comment','Energy1')
    dna_energy2=filter_spectra(dna_adducts,'comment','Energy2')
    
    put_energy0=filter_spectra(put_adducts,'comment','Energy0')
    put_energy1=filter_spectra(put_adducts,'comment','Energy1')
    put_energy2=filter_spectra(put_adducts,'comment','Energy2')
    
    smile_list=[]
    for i in range(len(dna_energy0)):
        smiles=dna_energy0[i].metadata.get('smiles/inchi')
        smile_list.append(smiles)
    
    smile_list_put=[]
    for i in range(len(put_energy0)):
        smiles=put_energy0[i].metadata.get('smiles/inchi')
        smile_list_put.append(smiles)
    ### MCS based nucleotide assignment to adducts
    
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    
    # Your dictionary of SMILES
    # C1C(C(OC1N2C=NC3=C(N=C(N=C32)N)N)CO)O
    # CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)O
    # C1C(C(OC1N2C=NC3=C2NC(=NC3=O)N)CO)O
    # C1C(C(OC1N2C=CC(=NC2=O)N)CO)O
    
    
    
    new_smiles_dict = {
        #'dA':'N2C=NC3=C(N=C(N=C32)N)N','dT': 'CC1=CN(C(=O)NC1=O)', 'dG': 'NC1=NC2=C(N=CN2)C(=O)N1', 'dC':'N2C=CC(=NC2=O)N'
        "dA": 'C1C(C(OC1N2C=NC3=C(N=C(N=C32)N)N)CO)O', #"OCC1OC(CC1O)n1cnc2c1ncnc2N",
        "dT": 'CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)O', #"OC1CC(OC1C)n1cc(C)c(=O)[nH]c1=O",
        "dG": 'NC1=NC2=C(N=CN2[C@H]2C[C@H](O)[C@@H](CO)O2)C(=O)N1', # "OCC1OC(CC1O)n1cnc2c1nc(N)[nH]c2=O",
        "dC": 'C1C(C(OC1N2C=CC(=NC2=O)N)CO)O', #"OCC1OC(CC1O)n1ccc(nc1=O)N",
    }
    # new_smiles_dict={'pu':"OCC1OC(CC1O)n1cnc2c1ncnc2",
    #     'py':"OCC1OC(CC1O)N1CC=CN=C1"}
    # Sample DataFrame with 'smiles' column
    data={'smiles':smile_list}
    data_p={'smiles_p':smile_list_put}
    df = pd.DataFrame(data)
    df_p = pd.DataFrame(data_p)
    # Function to convert SMILES to RDKit molecules
    def smiles_to_mol(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except:
            return None
    
    # Convert the 'smiles' column in the DataFrame to RDKit molecules
    df['mol'] = df['smiles'].apply(smiles_to_mol)
    df_p['mol'] = df_p['smiles_p'].apply(smiles_to_mol)
    # Remove any rows with invalid SMILES
    df = df.dropna(subset=['mol'])
    df_p = df_p.dropna(subset=['mol'])
    # Find the SMILES string in the dictionary with the largest maximum common substructure
    
    def find_largest_common_substructure(mol_list):
        max_common_substructure_size = 0
        largest_common_substructure = None
    
        for key, smiles_dict in new_smiles_dict.items():
            mol_dict = smiles_to_mol(smiles_dict)
            if mol_dict is not None:
                mcs = rdFMCS.FindMCS([mol_dict, mol_list])
                common_substructure_size = len(Chem.MolFromSmarts(mcs.smartsString).GetAtoms())
                if common_substructure_size > max_common_substructure_size:
                    max_common_substructure_size = common_substructure_size
                    largest_common_substructure = key
    
        return largest_common_substructure
    
    # function to find the largest common substructure to each molecule in the DataFrame
    df['largest_common_substructure'] = df['mol'].apply(find_largest_common_substructure)
    df_p['largest_common_substructure'] = df_p['mol'].apply(find_largest_common_substructure)
    # SMILES string with the largest maximum common substructure from the dictionary for each row
    df['largest_common_substructure_smiles'] = df['largest_common_substructure'].map(new_smiles_dict)
    df_p['largest_common_substructure_smiles'] = df_p['largest_common_substructure'].map(new_smiles_dict)
    # # Calculate the average size of molecules
    # avg_molecule_size = df['mol'].apply(lambda mol: mol.GetNumHeavyAtoms())
    
    # Calculate the percent similarity for each row
    df['atom_count_of_mcs_smarts'] = df.apply(
        lambda row: len(Chem.MolFromSmarts(rdFMCS.FindMCS([smiles_to_mol(new_smiles_dict[row['largest_common_substructure']]), row['mol']]).smartsString).GetAtoms()),axis=1)
    df_p['atom_count_of_mcs_smarts'] = df_p.apply(
        lambda row: len(Chem.MolFromSmarts(rdFMCS.FindMCS([smiles_to_mol(new_smiles_dict[row['largest_common_substructure']]), row['mol']]).smartsString).GetAtoms()),axis=1)
    
    #df['percent_similarity'] = (df['max_common_substructure_size'] / avg_molecule_size) * 100
    
    # Print the DataFrame with the results
    
    
    # In[58]:

    
    #metadata annotation
    for i in range(279):
        dna_energy0[i].set('nucleobase',df.largest_common_substructure[i])
        dna_energy1[i].set('nucleobase',df.largest_common_substructure[i])
        dna_energy2[i].set('nucleobase',df.largest_common_substructure[i])
    
    for i in range(303):
        put_energy0[i].set('nucleobase',df_p.largest_common_substructure[i])
        put_energy1[i].set('nucleobase',df_p.largest_common_substructure[i])
        put_energy2[i].set('nucleobase',df_p.largest_common_substructure[i])
    
    
    
    #purine and pyramidine bins for each energy level
    A0=filter_spectra(dna_energy0,'nucleobase','dA')
    T0=filter_spectra(dna_energy0,'nucleobase','dT')
    G0=filter_spectra(dna_energy0,'nucleobase','dG')
    C0=filter_spectra(dna_energy0,'nucleobase','dC')
    
    A0_p=filter_spectra(put_energy0,'nucleobase','dA')
    T0_p=filter_spectra(put_energy0,'nucleobase','dT')
    G0_p=filter_spectra(put_energy0,'nucleobase','dG')
    C0_p=filter_spectra(put_energy0,'nucleobase','dC')
    
    ##bins
    purine_energy0=A0+G0
    
    purine_energy0_p=A0_p+G0_p
    
    # In[63]:
    
    #py bins
    pyramidine_energy0=T0+C0
    
    pyramidine_energy0_p=T0_p+C0_p
    
    
    
    
    A1=filter_spectra(dna_energy1,'nucleobase','dA')
    T1=filter_spectra(dna_energy1,'nucleobase','dT')
    G1=filter_spectra(dna_energy1,'nucleobase','dG')
    C1=filter_spectra(dna_energy1,'nucleobase','dC')

    A1_p=filter_spectra(put_energy1,'nucleobase','dA')
    T1_p=filter_spectra(put_energy1,'nucleobase','dT')
    G1_p=filter_spectra(put_energy1,'nucleobase','dG')
    C1_p=filter_spectra(put_energy1,'nucleobase','dC')
    pyramidine_energy1=T1+C1
    purine_energy1=A1+G1
    
    pyramidine_energy1_p=T1_p+C1_p
    purine_energy1_p=A1_p+G1_p
    # In[65]:
    
    
    A2=filter_spectra(dna_energy2,'nucleobase','dA')
    T2=filter_spectra(dna_energy2,'nucleobase','dT')
    G2=filter_spectra(dna_energy2,'nucleobase','dG')
    C2=filter_spectra(dna_energy2,'nucleobase','dC')
    
    A2_p=filter_spectra(put_energy2,'nucleobase','dA')
    T2_p=filter_spectra(put_energy2,'nucleobase','dT')
    G2_p=filter_spectra(put_energy2,'nucleobase','dG')
    C2_p=filter_spectra(put_energy2,'nucleobase','dC')
    pyramidine_energy2=T2+C2
    purine_energy2=A2+G2
    pyramidine_energy2_p=T2_p+C2_p
    purine_energy2_p=A2_p+G2_p
    
    
    # In[66]:
    
    
    # #Metadata annotation
    for i in range(len(seed_a)):
        seed_a[i].set('id', f'a{i}')
        
    for i in range(len(seed_t)):
        seed_t[i].set('id', f't{i}')
    
    for i in range(len(seed_g)):
        seed_g[i].set('id', f'g{i}')
    
    for i in range(len(seed_c)):
        seed_c[i].set('id', f'c{i}')
    
    
      
    
    #generated smiles bin 
    purine=seed_a+seed_g
    pyramidine=seed_t+seed_c
      
    
    pur_lib_energy0=filter_spectra(purine,'comment','Energy0')
    pur_lib_energy1=filter_spectra(purine,'comment','Energy1')
    pur_lib_energy2=filter_spectra(purine,'comment','Energy2')
    pyr_lib_energy0=filter_spectra(pyramidine,'comment','Energy0')
    pyr_lib_energy1=filter_spectra(pyramidine,'comment','Energy1')
    pyr_lib_energy2=filter_spectra(pyramidine,'comment','Energy2')
    

    
    
    
    
    # In[71]:
    
    print('***calculating cosine similarity***')
    df_purine_energy0=match_spectra(purine_energy0,pur_lib_energy0)
    df_pyramidine_energy0=match_spectra(pyramidine_energy0,pyr_lib_energy0)
    
    df_purine_energy0_p=match_spectra(purine_energy0_p,pur_lib_energy0)
    df_pyramidine_energy0_p=match_spectra(pyramidine_energy0_p,pyr_lib_energy0)
    
    # In[72]:
    
    
    df_pyramidine_energy0
    
    
    # In[73]:
    
    print('***calculating cosine similarity***')
    df_purine_energy1=match_spectra(purine_energy1,pur_lib_energy1)
    df_pyramidine_energy1=match_spectra(pyramidine_energy1,pyr_lib_energy1)
    df_purine_energy1_p=match_spectra(purine_energy1_p,pur_lib_energy1)
    df_pyramidine_energy1_p=match_spectra(pyramidine_energy1_p,pyr_lib_energy1)
    
    # In[74]:
    
    print('***calculating cosine similarity***')
    df_purine_energy2=match_spectra(purine_energy2,pur_lib_energy2)
    df_pyramidine_energy2=match_spectra(pyramidine_energy2,pyr_lib_energy2)
    
    df_purine_energy2_p=match_spectra(purine_energy2_p,pur_lib_energy2)
    df_pyramidine_energy2_p=match_spectra(pyramidine_energy2_p,pyr_lib_energy2)
    
    # In[75]:
    
    
    df_e0=pd.concat([df_purine_energy0,df_pyramidine_energy0])
    df_e0.rename(columns={'ref_Smiles/Inchi': 'E0_ref_smile'}, inplace=True)
    df_e0.rename(columns={'Score':'Score_E0'}, inplace=True)
    
    df_e0_p=pd.concat([df_purine_energy0_p,df_pyramidine_energy0_p])
    df_e0_p.rename(columns={'ref_Smiles/Inchi': 'E0_ref_smile'}, inplace=True)
    df_e0_p.rename(columns={'Score':'Score_E0'}, inplace=True)
    
    # In[76]:
    
    
    df_e1=pd.concat([df_purine_energy1,df_pyramidine_energy1])
    df_e1.rename(columns={'ref_Smiles/Inchi': 'E1_ref_smile'}, inplace=True)
    df_e1.rename(columns={'Score':'Score_E1'}, inplace=True)
    
    df_e1_p=pd.concat([df_purine_energy1_p,df_pyramidine_energy1_p])
    df_e1_p.rename(columns={'ref_Smiles/Inchi': 'E1_ref_smile'}, inplace=True)
    df_e1_p.rename(columns={'Score':'Score_E1'}, inplace=True)
    
    
    # In[77]:
    
    
    df_e2=pd.concat([df_purine_energy2,df_pyramidine_energy2])
    df_e2.rename(columns={'ref_Smiles/Inchi': 'E2_ref_smile'}, inplace=True)
    df_e2.rename(columns={'Score':'Score_E2'}, inplace=True)
    
    df_e2_p=pd.concat([df_purine_energy2_p,df_pyramidine_energy2_p])
    df_e2_p.rename(columns={'ref_Smiles/Inchi': 'E2_ref_smile'}, inplace=True)
    df_e2_p.rename(columns={'Score':'Score_E2'}, inplace=True)
    
    # In[78]:
    
    print('***saving similarity matrix***')
    df_comb=pd.concat([df_e0['Query'],df_e0['Query_Smile'],df_e0['Score_E0'], df_e0['E0_ref_smile'],df_e1['Score_E1'], df_e1['E1_ref_smile'],df_e2['Score_E2'], df_e2['E2_ref_smile']],axis=1)
    df_comb.to_csv(out_adducts_csv_path)
    
    print('***saving similarity matrix***')
    df_comb_p=pd.concat([df_e0_p['Query'],df_e0_p['Query_Smile'],df_e0_p['Score_E0'], df_e0_p['E0_ref_smile'],df_e1_p['Score_E1'], df_e1_p['E1_ref_smile'],df_e2_p['Score_E2'], df_e2_p['E2_ref_smile']],axis=1)
    df_comb_p.to_csv(out_put_csv_path)
    
    # In[79]:
    
    
    df_comb['Score_E0'] = df_comb['Score_E0'].astype(float)
    df_comb['Score_E1'] = df_comb['Score_E1'].astype(float)
    df_comb['Score_E2'] = df_comb['Score_E2'].astype(float)
    
    df_comb_p['Score_E0'] = df_comb_p['Score_E0'].astype(float)
    df_comb_p['Score_E1'] = df_comb_p['Score_E1'].astype(float)
    df_comb_p['Score_E2'] = df_comb_p['Score_E2'].astype(float)
    
    
    
    import pandas as pd 
    df=pd.read_csv(out_adducts_csv_path)
    df['Score_E0'] = df['Score_E0'].astype(float)
    df['Score_E1'] = df['Score_E1'].astype(float)
    df['Score_E2'] = df['Score_E2'].astype(float)

    df_p=pd.read_csv(out_put_csv_path)
    df_p['Score_E0'] = df_p['Score_E0'].astype(float)
    df_p['Score_E1'] = df_p['Score_E1'].astype(float)
    df_p['Score_E2'] = df_p['Score_E2'].astype(float)
    
    
    # In[ ]:
    
    
    
    
    
    # In[82]:
    
    
    df.reset_index(drop=True, inplace=True)
    df_p.reset_index(drop=True, inplace=True)
    
    print('***Saving Plots***')
    
    # Set 'sno' column as the index (optional, depending on your data structure)
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Drop non-numeric columns from the DataFrame
    df_numeric = df.drop(['Unnamed: 0','Query', 'Query_Smile','E0_ref_smile','E1_ref_smile','E2_ref_smile'], axis=1)
    # Create a pivot table to reshape the DataFrame with 'sno' as rows and the numeric columns as columns
    heatmap_data = df_numeric
    c=['Score_E0','Score_E1','Score_E2']
    heatmap_data=heatmap_data[c]
    # Create the heatmap using seaborn
    sns.clustermap(heatmap_data[c], cmap='viridis', annot=False,col_cluster=False,figsize=(10, 8),dendrogram_ratio=(.1, .1),yticklabels=True,cbar_pos=(-.1, .2, .03, .4), fmt=".4f", cbar_kws={'label': 'Cosine Score'})
    plt.savefig(image_path + '/heatmap_adducts.png')
    # Show the plot
    plt.show()

    df_numeric = df_p.drop(['Unnamed: 0','Query', 'Query_Smile','E0_ref_smile','E1_ref_smile','E2_ref_smile'], axis=1)
    # Create a pivot table to reshape the DataFrame with 'sno' as rows and the numeric columns as columns
    heatmap_data = df_numeric
    c=['Score_E0','Score_E1','Score_E2']
    heatmap_data=heatmap_data[c]
    # Create the heatmap using seaborn
    sns.clustermap(heatmap_data[c], cmap='viridis', annot=False,col_cluster=False,figsize=(10, 8),dendrogram_ratio=(.1, .1),yticklabels=True,cbar_pos=(-.1, .2, .03, .4), fmt=".4f", cbar_kws={'label': 'Cosine Score'})
    plt.savefig(image_path + '/heatmap_putative_adducts.png')
    # Show the plot
    plt.show()
##histogram
    
    
    data = pd.read_csv(out_adducts_csv_path)
    thresholds = [0.7, 0.8, 0.9, 1.0]
    
    # Create lists to store the frequencies for each score and threshold
    frequencies_E0 = []
    frequencies_E1 = []
    frequencies_E2 = []
    
    # Calculate frequencies for each score and threshold
    for threshold in thresholds:
        filtered_data_E0 = data[data['Score_E0'] >= threshold]
        filtered_data_E1 = data[data['Score_E1'] >= threshold]
        filtered_data_E2 = data[data['Score_E2'] >= threshold]
        
        frequencies_E0.append(len(filtered_data_E0))
        frequencies_E1.append(len(filtered_data_E1))
        frequencies_E2.append(len(filtered_data_E2))
    
    # Create Seaborn histograms with threshold values on the x-axis and frequencies on the y-axis
    plt.figure(figsize=(12, 6))
    
    positions = range(len(thresholds))
    bar_width = 0.2  # Adjust this value for the desired bar separation
    
    # Create separate bar plots for each score column
    plt.bar([pos - bar_width for pos in positions], frequencies_E0, bar_width, label='Score_E0', color='skyblue')
    plt.bar(positions, frequencies_E1, bar_width, label='Score_E1', color='lightgreen')
    plt.bar([pos + bar_width for pos in positions], frequencies_E2, bar_width, label='Score_E2', color='coral')
    
    # Manually set the positions and labels of x-axis ticks
    plt.xticks(positions, [str(threshold) for threshold in thresholds])
    
    plt.xlabel("Threshold")
    plt.ylabel("Frequency")
    plt.title("Histograms of Scores Frequencies by Threshold for DNA Adducts")
    plt.legend()
    plt.tight_layout()
    
    # Specify the file path where you want to save the PNG file
    file_path = image_path + "/scores_threshold_DNA_adducts.png"
    plt.savefig(file_path, dpi=300)
    plt.show()
    
    print("bar plots for all three scores saved as a PNG file.")

    data = pd.read_csv(out_put_csv_path)
    thresholds = [0.7, 0.8, 0.9, 1.0]
    
    # Create lists to store the frequencies for each score and threshold
    frequencies_E0 = []
    frequencies_E1 = []
    frequencies_E2 = []
    
    # Calculate frequencies for each score and threshold
    for threshold in thresholds:
        filtered_data_E0 = data[data['Score_E0'] >= threshold]
        filtered_data_E1 = data[data['Score_E1'] >= threshold]
        filtered_data_E2 = data[data['Score_E2'] >= threshold]
        
        frequencies_E0.append(len(filtered_data_E0))
        frequencies_E1.append(len(filtered_data_E1))
        frequencies_E2.append(len(filtered_data_E2))
    
    # Create Seaborn histograms with threshold values on the x-axis and frequencies on the y-axis
    plt.figure(figsize=(12, 6))
    
    positions = range(len(thresholds))
    bar_width = 0.2  # Adjust this value for the desired bar separation
    
    # Create separate bar plots for each score column
    plt.bar([pos - bar_width for pos in positions], frequencies_E0, bar_width, label='Score_E0', color='skyblue')
    plt.bar(positions, frequencies_E1, bar_width, label='Score_E1', color='lightgreen')
    plt.bar([pos + bar_width for pos in positions], frequencies_E2, bar_width, label='Score_E2', color='coral')
    
    # Manually set the positions and labels of x-axis ticks
    plt.xticks(positions, [str(threshold) for threshold in thresholds])
    
    plt.xlabel("Threshold")
    plt.ylabel("Frequency")
    plt.title("Bar plots of Scores Frequencies by Threshold for putative Adducts")
    plt.legend()
    plt.tight_layout()
    
    # Specify the file path where you want to save the PNG file
    file_path = image_path + "/scores_threshold_putative_DNA_adducts.png"
    plt.savefig(file_path, dpi=300)
    plt.show()
    
    print("Barplots for all three scores saved as a PNG file.")
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Load your CSV file into a DataFrame
    data = pd.read_csv(out_adducts_csv_path)
    
    # Define the scores you want to plot
    scores_to_plot = ['Score_E0', 'Score_E1', 'Score_E2']
    
    # Set the number of bins for the histogram
    num_bins = 20
    # Create separate histograms for each score
    colors = ['skyblue', 'lightgreen', 'coral']
    
    # Create separate histograms for each score with custom colors
    for i, score_column in enumerate(scores_to_plot):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=score_column, bins=num_bins, kde=True, color=colors[i])
    
    
        plt.xlabel(score_column)
        plt.ylabel("Frequency")
        plt.title(f"{score_column} -Distribution of cosine similarity: Generated adducts vs DNA Adducts")
        plt.grid(True)
        
        # Specify the file path where you want to save the PNG file
        file_path = image_path + f"/{score_column}_adducts_histogram.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    print("Histograms for individual scores saved as PNG files.")    

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Load your CSV file into a DataFrame
    data = pd.read_csv(out_put_csv_path)
    
    # Define the scores you want to plot
    scores_to_plot = ['Score_E0', 'Score_E1', 'Score_E2']
    
    # Set the number of bins for the histogram
    num_bins = 20
    # Create separate histograms for each score
    colors = ['skyblue', 'lightgreen', 'coral']
    
    # Create separate histograms for each score with custom colors
    for i, score_column in enumerate(scores_to_plot):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x=score_column, bins=num_bins, kde=True, color=colors[i])
    
    
        plt.xlabel(score_column)
        plt.ylabel("Frequency")
        plt.title(f"{score_column} -Distribution of cosine similarity: Generated adducts vs Non-Bonafide (putative) Adducts")
        plt.grid(True)
        
        # Specify the file path where you want to save the PNG file
        file_path = image_path + f"/{score_column}_putative_histogram.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    print("Histograms for individual scores saved as PNG files.")  


#exp_directory = '/home/shivaji/trail/result/csv/txt/cfm'
#put_directory = '/home/shivaji/trail/result/csv/txt/cfm'
#out_directory = '/home/shivaji/trail/src'
#image_path = '/home/shivaji/trail/result'
#
#A_path = os.path.join(out_directory, 'A.msp')
#T_path = os.path.join(out_directory, 'T.msp')
#G_path = os.path.join(out_directory, 'G.msp')
#C_path = os.path.join(out_directory, 'C.msp')
#exp_279_path = os.path.join(exp_directory, 'exp_279.msp')
#putative_path = os.path.join(put_directory, 'putative.msp')
#out_adducts_csv_path = os.path.join(out_directory, 'output_adducts.csv')
#out_put_csv_path = os.path.join(out_directory, 'output_put.csv')
#
#
#spectra_viz(A_path, T_path, G_path, C_path, exp_279_path, putative_path, out_adducts_csv_path, out_put_csv_path, image_path) 
#