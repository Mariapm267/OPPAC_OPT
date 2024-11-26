import sys
sys.path.append('../')
import regressor_utils   

version = '1step'
file = f'../../processed_datasets/processed_data_{version}.pickle'

regressor_utils.split_data(file, dest_folder = '../../split_processed_datasets' + '_' + version , version = version)
