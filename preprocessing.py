import numpy as np
from scipy.optimize import curve_fit
import os

class process_events():
    def __init__(self):
        pass
      
    def get_data_from_files(directory, n_steps = 1):
        if n_steps not in [1, 2]:
            raise ValueError("n_steps debe ser 1 o 2")   
        
        def get_inputs(file):     # get simulation conditions (regressor inputs) from file's name
            params =  file.split("_")
            dcol = float(params[1][3:])
            p = float(params[2][1:])
            x = float(params[3][1:])
            y = float(params[4][1:])
            return [p, dcol, x, y]
        
        def gaussian_fit(x, y):
            def gaussian(x, mu, sigma, A):
                return A * np.exp(-0.5 * ((x - mu) / sigma)**2)
            p0=[np.mean(x), np.std(x), np.max(y)]
            popt, _ = curve_fit(gaussian, x, y, p0)
            mu_fit, sigma_fit, _ = popt
            return mu_fit, sigma_fit

        def get_array_stats(array):    # por cada array de SiPMs, obtiene N, mu, sigma
            N = sum(array)                      
            L = np.arange(0, len(N))   # number of SiPM
            mu, sigma = gaussian_fit(L, array)
            return N, mu, sigma

        def convertir_unidades(values):
            interp_values = []
            for value in values:
                interp_values.append(np.interp(value, (0, 33), (-10000, 10000)))
            return interp_values
    
        def weighted_mean(output):
            Nx1, mux1, sigmax1, Nx2, mux2, sigmax2, Ny1, muy1, sigmay1, Ny2, muy2, sigmay2 = output
            x_hat = (mux1*Nx1/sigmax1 + mux2*Nx2/sigmax2 ) / (Nx1/sigmax1 + Nx2/sigmax2)
            y_hat = (muy1*Ny1/sigmay1 + muy2*Ny2/sigmay2 ) / (Ny1/sigmay1 + Ny2/sigmay2)
            return [x_hat,y_hat]
    
        def get_target(file, n_steps):
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
                        mu, sigma = convertir_unidades((mu, sigma))
                        output_array.append(N); output_array.append(mu); output_array.append(sigma)
                    if n_steps == 2:
                        targets.append(output_array)
                    else:
                        targets.append(weighted_mean(output_array))
            return targets
    
        targets = []
        inputs = []
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            input = get_inputs(path)
            targets = get_target(path,n_steps)
            for i in range(len(targets)):
                inputs.append(input)
                targets.append(targets[i])
        return inputs, targets
