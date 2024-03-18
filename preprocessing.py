import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import pickle

    
def get_data_from_files(directory, output_file, n_steps = 1):
    '''

    Writes a pickle file with simulation conditions versus final conditions (reconstructed possition (1 step) or photon distribution stats (2 steps))
  
    Input: 
        - directory: dir with simulation files
        - output_file: where the processed data is written
        - n_steps: 
            - 1 step : the output is the resontructed possition of the alpha particle
            - 2 steps: the output is the N, mu, sigma of the photon distribution on each 'wall'
  
    '''
    if n_steps not in [1, 2]:
        raise ValueError("n_steps debe ser 1 o 2")   
    
    def get_inputs(file):    
        '''
        Inputs : file name like: datasets/simu_col5.0_p10.0_x-1000.0_y-1000.0
        Outputs : simulation conditions (dcol, preassure, xbeam, ybeam)
        '''
        params =  file.split("_")
        dcol = float(params[1][3:])
        p = float(params[2][1:])
        x = float(params[3][1:])
        y = float(params[4][1:])
        return [p, dcol, x, y]
    
    def gaussian_fit(x, y):
        '''
        Inputs : x, y
        Outputs : mu, sigma from gaussian fit y(x)
        '''
        def gaussian(x, mu, sigma, A):
            return A * np.exp(-0.5 * ((x - mu) / sigma)**2)
        p0=[np.mean(x), np.std(x), np.max(y)]
        popt, _ = curve_fit(gaussian, x, y, p0)
        mu_fit, sigma_fit, _ = popt
        return mu_fit, sigma_fit
    
    def distance_interp(values):
        '''
        Input: distance array (mu or sigma) in SiPM dimensions  (from 0 to 33)
        
        Output:  array in distance dimensions ( from -5000 to 5000)
        '''
        interp_values = []
        for value in values:
            interp_values.append(np.interp(value, (0, 33), (-5000, 5000)))
        return interp_values

    def get_array_stats(array):    # por cada array de SiPMs, obtiene N, mu, sigma
        '''
        Inputs: array of detected photons in each SiPM
        
        Outputs:
            - N: total number of photons
            - mu: mean possition () from a gaussian fit [ in range (-5000, 5000)]
            - sigma: uncertainty from a gaussian fit [ in range (-5000, 5000)]
        '''
        N = sum(array)                      
        L = np.arange(1,34)                          # enumerate  33 SiPM from 1 to 33
        mu, sigma = gaussian_fit(L, array)
        mu, sigma = distance_interp((mu, sigma))     # interpolation to get distance units
        return N, mu, sigma

   
    def weighted_mean(output):
        '''
        Input: 4 SiPMs arrays stats
        Output: estimated possition of charged particle (weighted mean formula)
        '''
        Nx1, mux1, sigmax1, Nx2, mux2, sigmax2, Ny1, muy1, sigmay1, Ny2, muy2, sigmay2 = output
        x_hat = (mux1*Nx1/sigmax1 + mux2*Nx2/sigmax2 ) / (Nx1/sigmax1 + Nx2/sigmax2)
        y_hat = (muy1*Ny1/sigmay1 + muy2*Ny2/sigmay2 ) / (Ny1/sigmay1 + Ny2/sigmay2)
        return [x_hat,y_hat]

    def get_target(file, n_steps):
        '''
        Input: file, nsteps (number of steps in particle possition reconstruction (1 or 2))
        Output: list of targets
            - If nsteps = 1, targets are x_hat and y_hat. i.e. reconstructed possition
            - If nsteps = 2, targets are N, mu, sigma for the 4 arrays of SiPms in each event
        '''
        targets = []
        with open(file, 'r') as archivo:
            lines = archivo.readlines()
            for line in lines:
                values = line.strip().split()   
                values = values[:-1]              
                event= [float(value) for value in values]
                output_array = []   
                for array in np.array_split(event, 4):     
                    N, mu, sigma = get_array_stats(array)
                    output_array.append(N); output_array.append(mu); output_array.append(sigma)
                if n_steps == 2:
                    targets.append(output_array)
                else:
                    targets.append(weighted_mean(output_array))
        return targets

    targets_total = []
    inputs_total = []
    for file in os.listdir(directory): 
        path = os.path.join(directory, file)
        input = get_inputs(path)
        targets = get_target(path,n_steps)
        for i in range(len(targets)):
            inputs_total.append(input)
            targets_total.append(targets[i])
            
    dcol, p, x, y = zip(*inputs_total)
    if n_steps == 1:
        x_hat, y_hat = zip(*targets_total)
        data = {'dcol': dcol, 'p': p, 'x': x, 'y': y, 'x_hat': x_hat, 'y_hat': y_hat}
        
    else: 
        Nx1, mux1, sigmax1, Nx2, mux2, sigmax2, Ny1, muy1, sigmay1, Ny2, muy2, sigmay2 = zip(*targets_total)
        data = {'dcol': dcol, 'p': p, 'x': x, 'y': y, 
                 'Nx1': Nx1, 'mux1': mux1, 'sigmax1': sigmax1, 
                 'Nx2': Nx2, 'mux2': mux2, 'sigmax2': sigmax2,
                 'Ny1': Ny1, 'muy1': muy1, 'sigmay1': sigmay1,
                 'Ny2': Ny2, 'muy2': muy2, 'sigmay2': sigmay2}

    # Save data in a .pickle file
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)
    
    return None
