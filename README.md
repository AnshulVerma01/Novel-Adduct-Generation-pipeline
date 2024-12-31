# Novel Adduct Generation Pipeline

Genotoxins can form DNA adducts by covalently binding to cellular DNA, leading to structural and functional changes that disrupt cellular processes. These disruptions can occur through genetic mechanisms (e.g., mutations) or non-genetic mechanisms (e.g., epigenetic modifications). The identification, quantification, and characterization of DNA adducts are critical for understanding their biological impact and can accelerate efforts to predict carcinogenicity and elucidate their modes of action.

This pipeline leverages computational strategies, including Artificial Intelligence (AI)-based modeling, to advance the study of DNA adducts. These approaches enable predictive modeling for adduct formation via metabolic activation and the identification of novel adducts, providing valuable insights into their biological significance and role in genotoxicity.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AnshulVerma01/Novel-adducts-generation.git
```
2. Create a conda environment:
```bash
conda env create -f adductGen.yml
conda activate adductGen
```

## Usage
The primary script to run the pipeline is provided src/mutAIverse.py. Execute it directly by changing the respective file paths.
```python
python mutAIverse.py
```

- In the result/csv/txt/tanimoto/adducts directory, we have provided experimentally validated and putative adducts for each nucleotide. To generate the Tanimoto similarity plot, execute the following command:
```python
python tanimoto.py
```
This will generate a Tanimoto similarity plot based on the provided adduct data.
