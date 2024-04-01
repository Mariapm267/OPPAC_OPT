import numpy as np
import pandas as pd
#from scipy.optimize import curve_fit
import os
import pickle
import tqdm

def get_data_from_files(directory, output_file, n_steps = 1):
    if n_steps not in [1, 2]:
        raise ValueError("n_steps debe ser 1 o 2")   
    
    def get_inputs(file):     # condiciones de la simulacion a partir del nombre del archivo
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
    
    def distance_interp(array):
        '''
        Input: distance array in SiPM dimensions  (from 0 to 33)
        Output:  array in distance dimensions ( from -5000 to 5000)
        '''
        return [np.interp(value, (0, 33), (-5000, 5000)) for value in array]

    def gaussian(x, mu, sigma, A):
            return A * np.exp(-0.5 * ((x - mu) / sigma)**2)
        
    def get_array_stats(array):   
        '''
        Inputs: array of detected photons in each SiPM
        Outputs:
            - N: total number of photons
            - mu: mean possition () from a gaussian fit (or mean and std if curve_fit fails) [ in range (-5000, 5000)]
            - sigma: uncertainty from a gaussian fit (or mean and std if curve_fit fails) [ in range (-5000, 5000)]
            - discarded event: true or false (if no photons detected, the event is discarded)
        '''
        N = sum(array)                      
        bins = np.arange(0,34)                          # enumerate  33 SiPM from 1 to 33  
        midbins = (bins[:-1] + bins[1:]) / 2
        midbins = distance_interp(midbins)              # interpolate to get distance units 
          
        if N == 0:
            mean, std= [9999, 9999]   #valor por defecto
            discarded_event = True  
            
        else: 
            discarded_event = False
            mean = np.sum(array * midbins) / N
            std = np.sqrt(np.sum(array * ((midbins - mean) ** 2)) / N) 
            if std == 0:
                std =  (10000/33)/np.sqrt(12)    # if std = 0, we assign the minimum sigma for our setting, which is the length of the PM /sqrt(12)
            #with warnings.catch_warnings():
            #    warnings.simplefilter("error", OptimizeWarning) 
            #    try: 
            #        p0=[mean, std, np.max(array)]   
            #        popt, _ = curve_fit(gaussian, midbins, array, p0 = p0)
            #        mu, sigma, _ = popt
            #        discarded_event = False 

            #    except (RuntimeWarning, RuntimeError, OptimizeWarning):           # se non converxe que calcule a mediana e a std
            #        mu = mean
            #        sigma = std
            #        discarded_event = False 
        return [N, mean, std, discarded_event]

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
        n = 0
        with open(file, 'r') as archivo:
            lines = archivo.readlines()
            for line in lines:
                n +=1
                values = line.strip().split()   
                values = values[:-1]              
                event= [float(value) for value in values]
                output_array = []   
                for array in np.array_split(event, 4):     
                    N, mu, sigma, discarded_event = get_array_stats(array)
                    output_array.append(N); output_array.append(mu); output_array.append(sigma)
                    if discarded_event == True:
                        break
                if discarded_event == False:     # falta algo para asignar un valor por defecto
                    if n_steps == 2:
                        targets.append(output_array)
                    else:
                        targets.append(weighted_mean(output_array))
        return targets


    with tqdm(total=len(os.listdir(directory))) as pbar:
        targets_total = []
        inputs_total = []
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            input = get_inputs(path)
            targets = get_target(path,n_steps)
            for i in range(len(targets)):
                inputs_total.append(input)
                targets_total.append(targets[i])
            pbar.update(1)
                
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
    