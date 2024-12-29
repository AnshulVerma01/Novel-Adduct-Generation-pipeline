import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

def cfmid(input, output_directory, es_mode='positive'):
    input_filename = os.path.basename(input)
    output_filename = os.path.splitext(input_filename)[0] + '.msp'

    if es_mode == 'negative':
        command = f'docker run --rm=true -v {os.getcwd()}:/cfmid/public/ -i wishartlab/cfmid:latest sh -c "cd /cfmid/public/; cfm-predict \'{input_filename}\' 0.001 /trained_models_cfmid4.0/[M-H]-/param_output.log /trained_models_cfmid4.0/[M-H]-/param_config.txt 1 \'{output_filename}\'"'
    else:
        command = f'docker run --rm=true -v {os.getcwd()}:/cfmid/public/ -i wishartlab/cfmid:latest sh -c "cd /cfmid/public/; cfm-predict \'{input_filename}\' 0.001 /trained_models_cfmid4.0/[M+H]+/param_output.log /trained_models_cfmid4.0/[M+H]+/param_config.txt 1 \'{output_filename}\'"'
    
    output_filepath = os.path.join(output_directory, output_filename)

    subprocess.run(command, shell=True)

def cfmid_predict(input_files, output_directory, es_mode='positive', num_threads=2):
    os.makedirs(output_directory, exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for input_file in input_files:
            executor.submit(cfmid, input_file, output_directory, es_mode)

# Usage:
# in_directory = '/home/shivaji/trail/src'
# output_directory = '/home/shivaji/trail/src'
# es_mode = 'positive'  # or 'negative' if needed
# num_threads = 1  # Specify the number of threads to use

# # List all text files in the input directory
# input_files = [os.path.join(in_directory, file) for file in os.listdir(in_directory) if file.endswith('.txt')]

# # Call process_files to run cfmid on different threads
# cfmid_predict(input_files, output_directory, es_mode, num_threads)




