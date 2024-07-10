import numpy as np
import pandas as pd

#time visualization
from tqdm import tqdm

#my modules
import sys
sys.path.append('../')
from preprocessing import get_data_from_files

directory = '../../Datasets'
output_folder = '../../processed_datasets/'

if not os.path.exists(output_folder):
		os.makedirs(output_folder)

get_data_from_files(directory, output_folder + 'processed_data.pickle', n_steps = 1)
