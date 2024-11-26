import os
import numpy as np
import pickle
from tqdm import tqdm

def get_data_from_files(directory, output_file, n_steps=1, nevents_per_file=10000, get_only_distributions=False):
    def get_inputs(file):  # simulation conditions from file name
        f = os.path.basename(file)
        params = f.split("_")
        dcol = float(params[1][3:])
        p = float(params[2][1:])
        x = float(params[3][1:])
        y = float(params[4][1:])
        return [dcol, p, x, y]

    def distance_interp(array):
        return [np.interp(value, (0, 33), (-5000, 5000)) for value in array]

    def get_array_stats(array):
        N = sum(array)                      
        bins = np.arange(0, 34)  # enumerate 33 SiPM from 1 to 33  
        midbins = (bins[:-1] + bins[1:]) / 2
        midbins = distance_interp(midbins)  # interpolate to get distance units 

        if N == 0:
            mean, std = [9999, 9999]  # valor por defecto
            discarded_event = True  
        else: 
            discarded_event = False
            mean = np.sum(array * midbins) / N
            std = np.sqrt(np.sum(array * ((midbins - mean) ** 2)) / N) 
            if std == 0:
                std = (10000 / 33) / np.sqrt(12)  # minimum sigma

        return [N, mean, std, discarded_event]

    def weighted_mean(output):
        Nx1, mux1, sigmax1, Nx2, mux2, sigmax2, Ny1, muy1, sigmay1, Ny2, muy2, sigmay2 = output
        x_hat = (mux1 * Nx1 / sigmax1 + mux2 * Nx2 / sigmax2) / (Nx1 / sigmax1 + Nx2 / sigmax2)
        y_hat = (muy1 * Ny1 / sigmay1 + muy2 * Ny2 / sigmay2) / (Ny1 / sigmay1 + Ny2 / sigmay2)
        return [x_hat, y_hat]

    def get_target(file, n_steps):
        targets = []
        with open(file, 'r') as archivo:
            for i in range(nevents_per_file):
                try:     
                    line = archivo.readline()
                    if not line:  
                        break
                    values = line.strip().split()   
                    values = values[:-1]              
                    event = [float(value) for value in values]
                    output_array = []  
                    for array in np.array_split(event, 4):     
                        N, mu, sigma, discarded_event = get_array_stats(array)
                        output_array.append(N)
                        output_array.append(mu)
                        output_array.append(sigma)
                        if discarded_event:
                            break
                    if not discarded_event: 
                        if n_steps == 2:
                            targets.append(output_array)
                        else:
                            targets.append(weighted_mean(output_array))
                except ValueError:
                    print('Failed event in file:', file)
        return targets

    def get_photon_distributions(file):
        events = []
        with open(file, 'r') as archivo:
            try:
                for i in range(nevents_per_file):
                    line = archivo.readline()
                    if not line: 
                        break
                    values = line.strip().split()   
                    values = values[:-1]  # Eliminar el último valor si es necesario              
                    event = [float(value) for value in values]
                    
                    # Descartar el evento si todos los valores son cero
                    if all(value == 0 for value in event):
                        continue
                    
                    events.append(event)
            except ValueError:
                print('Failed event in file:', file)
        return events


    targets_total = []
    inputs_total = []
    if get_only_distributions:
        print('Getting the raw events')
        with tqdm(total=len(os.listdir(directory))) as pbar:
            for file in os.listdir(directory):
                input_data = get_inputs(os.path.join(directory, file))
                targets = get_photon_distributions(os.path.join(directory, file))
                for i in range(len(targets)):
                    inputs_total.append(input_data)
                    targets_total.append(targets[i])

                pbar.update(1)

    else:
        targets_total = []
        inputs_total = []

        with tqdm(total=len(os.listdir(directory))) as pbar:
            for file in os.listdir(directory):
                input_data = get_inputs(os.path.join(directory, file))
                targets = get_target(os.path.join(directory, file), n_steps)

                # Iterar sobre los targets obtenidos
                for i in range(len(targets)):
                    inputs_total.append(input_data)
                    targets_total.append(targets[i])

                pbar.update(1)

    dcol, p, x, y = zip(*inputs_total)
    if get_only_distributions:
        # Si get_only_distributions es True, los targets son los valores de 'nphotons'
        photon_distributions = targets_total
        data = {'dcol': dcol, 'p': p, 'x': x, 'y': y, 'nphotons': photon_distributions}
    elif n_steps == 1:
        # Si n_steps es 1, los targets son x_hat e y_hat
        x_hat, y_hat = zip(*targets_total)
        data = {'dcol': dcol, 'p': p, 'x': x, 'y': y, 'x_hat': x_hat, 'y_hat': y_hat}
    else:
        # Si n_steps es 2, los targets son los datos completos de Nx, mu, sigma para cada SiPM
        Nx1, mux1, sigmax1, Nx2, mux2, sigmax2, Ny1, muy1, sigmay1, Ny2, muy2, sigmay2 = zip(*targets_total)
        data = {'dcol': dcol, 'p': p, 'x': x, 'y': y, 
                'Nx1': Nx1, 'mux1': mux1, 'sigmax1': sigmax1, 
                'Nx2': Nx2, 'mux2': mux2, 'sigmax2': sigmax2,
                'Ny1': Ny1, 'muy1': muy1, 'sigmay1': sigmay1,
                'Ny2': Ny2, 'muy2': muy2, 'sigmay2': sigmay2}

    # Guardar los datos en un solo archivo
    with open(output_file, 'wb') as file_out:  # Guardar todos los datos
        pickle.dump(data, file_out)

    # Mensaje final indicando el guardado
    print(f'Data saved to {output_file}')

