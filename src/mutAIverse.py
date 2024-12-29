 #export TFHUB_CACHE_DIR=./tmp


from biotransformer import *
from adduct_gen import *
#from cfmid import *
#from spectra_viz import *
#from tanimoto import *

iteration = 0 #[choices = 0,1,2,3,4]
output = "/home/shivaji/trail/data/test_output"
input_file = '/home/shivaji/trail/data/biot_input.csv'
txt_file = "/home/shivaji/trail/data/smiles.txt"   ####output file for biotransformation

##implementation of biotransfomer function
if iteration==0:
    biot_0_it(input_file, txt_file)
else:
    biot_many_it(input_file, output, iteration, txt_file)
     

mimosa(txt_file)



# in_directory = '/home/shivaji/trail/src'
# output_directory = '/home/shivaji/trail/src'
# es_mode = 'positive'  # or 'negative' if needed
# num_threads = 4  # Specify the number of threads to use

# # List all text files in the input directory
# input_files = [os.path.join(in_directory, file) for file in os.listdir(in_directory) if file.endswith('.txt')]

# # Call process_files to run cfmid on different threads
# cfmid_predict(input_files, output_directory, es_mode, num_threads)



# # Use dynamic path creation
# exp_directory = '/home/shivaji/trail/result/csv/txt/cfm'
# put_directory = '/home/shivaji/trail/result/csv/txt/cfm'
# out_directory = '/home/shivaji/trail/src'
# image_path = '/home/shivaji/trail/result'

# A_path = os.path.join(out_directory, 'A.msp')
# T_path = os.path.join(out_directory, 'T.msp')
# G_path = os.path.join(out_directory, 'G.msp')
# C_path = os.path.join(out_directory, 'C.msp')
# exp_279_path = os.path.join(exp_directory, 'exp_279.msp')
# putative_path = os.path.join(put_directory, 'putative.msp')
# out_adducts_csv_path = os.path.join(out_directory, 'output_adducts.csv')
# out_put_csv_path = os.path.join(out_directory, 'output_put.csv')


# spectra_viz(A_path, T_path, G_path, C_path, exp_279_path, putative_path, out_adducts_csv_path, out_put_csv_path, image_path) 



# input_dir = '/home/shivaji/trail/result/csv/txt/tanimoto_txt'
# exp_dir = '/home/shivaji/trail/result/csv/txt/tanimoto_txt/adducts'
# tanimoto_score(input_dir, exp_dir)

# input_dir = '/home/shivaji/trail/result/csv/txt/tanimoto_txt'
# adduct_dir = '/home/shivaji/trail/result/csv/txt/tanimoto_txt/adducts'
# tanimoto_heatmap(input_dir, adduct_dir)

