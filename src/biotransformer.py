import os
import subprocess
import csv
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from Metabokiller import mk_predictor as mk
from openbabel import openbabel  # You might need to install this package



def convert_to_canonical_smiles(smiles):
    # Initialize Open Babel
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "can")

    # Create an Open Babel molecule object from the SMILES string
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, smiles)

    # Convert to canonical SMILES
    canonical_smiles = obConversion.WriteString(mol).strip()

    return canonical_smiles


def parse_sdf_file(file_path):
    suppl = Chem.SDMolSupplier(file_path)
    field_dict = {}

    for mol in suppl:
        if mol is not None:
            props = mol.GetPropsAsDict()
            for prop_name, prop_value in props.items():
                if prop_name not in field_dict:
                    field_dict[prop_name] = [prop_value]
                else:
                    field_dict[prop_name].append(prop_value)

    return field_dict

def biotransformer(smiles, output_file_name, output_directory, command_number, iteration):
    '''
    Runs biotransformer for EC based, Cyp450 (phaseI), and phase II biotransformation reactions.
    '''
    if command_number == 1:
        biotransform_type = "cyp450"
    elif command_number == 2:
        biotransform_type = "phaseII"
    elif command_number == 3:
        biotransform_type = "ecbased"
    else:
        raise ValueError("Invalid command number")

    command = f'java -jar BioTransformer3.0_20230525.jar -k pred -b {biotransform_type} -ismi "{smiles}" -osdf "{output_directory}/{output_file_name}" -s {iteration} -cm 3'
    subprocess.run(command, shell=True)

def biot_many_it(input_file, output_directory, iteration, txt_file):
    #def convert_to_canonical_smiles(smiles):
        # Initialize Open Babel
    #    obConversion = openbabel.OBConversion()
    #    obConversion.SetInAndOutFormats("smi", "can")
    
        # Create an Open Babel molecule object from the SMILES string
    #    mol = openbabel.OBMol()
    #    obConversion.ReadString(mol, smiles)
    
        # Convert to canonical SMILES
    #    canonical_smiles = obConversion.WriteString(mol).strip()
    
    #    return canonical_smiles
    #    pass

    # Read the input file
    with open(input_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > 0:
                smiles = row[2].strip()
                file_name = row[1].strip()

                for i in range(1, 4):
                    output_file_name = f"{file_name}_command{i}.sdf"
                    biotransformer(smiles, output_file_name, output_directory, i, iteration)

    # Process SDF files and create a combined CSV
    field_values_dicts = []
    for file_name in os.listdir(output_directory):
        if file_name.endswith('.sdf'):
            file_path = os.path.join(output_directory, file_name)
            if os.path.getsize(file_path) == 0:
                print("Empty file. Skipping:", file_path)
                continue
            field_values_dict = parse_sdf_file(file_path)
            field_values_dicts.append(field_values_dict)

    combined_dict = {}
    for field_values_dict in field_values_dicts:
        for field, values in field_values_dict.items():
            if field not in combined_dict:
                combined_dict[field] = values
            else:
                combined_dict[field].extend(values)

    df = pd.DataFrame(combined_dict)
    df.to_csv('combined.csv', index=False)

    # Run Metabokiller
    df = pd.read_csv('combined.csv')
    df = df.dropna(subset=['SMILES'])
    df['SMILES'] = df['SMILES'].str.strip()
    df['Canonical_SMILES'] = df['SMILES'].apply(convert_to_canonical_smiles)
    df = df.drop_duplicates(subset=['Canonical_SMILES'])
    df.to_csv('combined_filtered.csv')
    mk_smiles = df['Canonical_SMILES'].to_list()

    chunk_size = 25000
    num_chunks = (len(mk_smiles) + chunk_size - 1) // chunk_size
    pbar = tqdm(total=num_chunks, desc="Processing Chunks")

    for i in range(0, len(mk_smiles), chunk_size):
        chunk = mk_smiles[i:i + chunk_size]
        res_el = mk.Electrophile(chunk)

        if i == 0:
            res_el.to_csv('mk_el.csv', index=False)
        else:
            res_el.to_csv('mk_el.csv', mode='a', header=False, index=False)

        pbar.update(1)

    pbar.close()

    for i in range(0, len(mk_smiles), chunk_size):
        chunk = mk_smiles[i:i + chunk_size]
        res_ox = mk.Oxidative(chunk)

        if i == 0:
            res_ox.to_csv('mk_ox.csv', index=False)
        else:
            res_ox.to_csv('mk_ox.csv', mode='a', header=False, index=False)

        pbar.update(1)

    pbar.close()
    #metabokiller predictions filtering
    ox_pred = pd.read_csv('mk_ox.csv')
    el_pred = pd.read_csv('mk_el.csv')
    mk_res=pd.DataFrame()
    mk_res['Electrophile_preds'] = el_pred['Electrophile_preds']
    mk_res['Oxidative_preds'] = ox_pred['Oxidative_preds']
    mk_res['smiles']=el_pred['smiles']
    #keeping either or both electrophilicity=1 and oxidative damage=1 predictions
    mk_selected = mk_res[(mk_res['Electrophile_preds'] != 0) | (mk_res['Oxidative_preds'] != 0)]
    mk_selected.to_csv('mk_selected.csv')
    #saving smiles for Mimosa
    mk_selected['smiles'].to_csv('smiles.txt', index=False)
    

def biot_0_it(input_file, txt_file):

    # Open the CSV file for reading and the output text file for writing
    with open(input_file, 'r') as csv_file, open(txt_file, 'w') as txt_file:
        # Create a CSV reader
        csv_reader = csv.reader(csv_file)

        # Iterate through the CSV rows
        for row in csv_reader:
            if len(row) >= 3:
                # Assuming the SMILES string is in the third column (index 2)
                smiles = row[2]
                # Write the SMILES to the output text file
                txt_file.write(smiles + '\n') 



# iteration = 1 #[choices = 0,1,2,3,4]
# output_directory = "/home/shivaji/trail/data/test_output"
# input_file = "/home/shivaji/trail/data/input.csv"

# ##implementation of biotransfomer function
# if iteration==0:
#     biot_0_it(input_file)
# else:
#     biot_many_it(input_file, output_directory, iteration)
