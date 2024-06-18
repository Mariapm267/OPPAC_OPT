import numpy as np
import pandas as pd

#time visualization
from tqdm import tqdm

#my modules
import sys
sys.path.append('../')
from preprocessing import get_data_from_files

directory = '/scratch04/maria.pereira/TFM/Datasets'
output_file = '../../processed_datasets/processed_data.pickle'
get_data_from_files(directory, output_file, n_steps = 1)