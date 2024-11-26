import sys
sys.path.append('../')
from preprocessing import get_data_from_files


directory = '/scratch04/maria.pereira/TFM/Datasets/'
version = '1step'   #'2steps' or '1step'
output_file = f'../../processed_datasets/processed_data_{version}.pickle'

get_data_from_files(directory, output_file, n_steps = None, nevents_per_file = 1000, get_only_distributions = True)



