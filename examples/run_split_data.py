import sys
sys.path.append('../')
import regressor_utils   


file = '../../processed_datasets/processed_data.pickle'
regressor_utils.split_data(file, dest_folder = '../../processed_datasets')