import subprocess
import numpy as np
from multiprocessing import Pool
import os
import time



def simular(args, simulation_executable = '../../OPPAC_Sim/build/exe_scinti'):
    '''
    This function receives the detector configuration and calls the G4 executable to simulate the events
    '''
    start_time = time.time() 
    n_events, d_col, output_name, pres, x_beam, y_beam = args
    for _ in range(n_events):
        subprocess.run([f"{simulation_executable}", "1", f'{d_col}', output_name, f'{pres}', f'{x_beam}', f'{y_beam}'])
    end_time = time.time()  
    total_time = end_time - start_time  
    return total_time

def main(presiones, ds_col, xs_beam, ys_beam, n_events = 10000, num_cores = 8):
    args_list = []    
    sim_not_done = 0
    for p in presiones:
        for d in ds_col:
            for x in xs_beam:
                for y in ys_beam:
                    output_name = f"/scratch04/maria.pereira/TFM/Datasets/simu_col{d}_p{p}_x{x}_y{y}"
                    if os.path.exists(output_name):  # Check if the file already exists
                        with open(output_name, 'r') as file:
                            lines = file.readlines()
                        if len(lines) < n_events:  # Check if the file has less than n_events lines (each line is an event)
                            remaining_events = n_events - len(lines)
                            args_list.append((remaining_events, d, output_name, p, x, y))
                            sim_not_done += 1
                            print('not completed')
                    else:
                        args_list.append((n_events, d, output_name, p, x, y))
                        sim_not_done += 1
                        
    total_simulations = len(presiones)*len(ds_col)*len(xs_beam)*len(ys_beam)

    with open("simulations_log.txt", "w") as log_file:
        log_file.write(f"Numero total de simulaciones: {total_simulations}\n")
        log_file.write(f"Total de simulaciones que quedan: {sim_not_done}\n")

    with Pool(num_cores) as pool:
        for sim_time in pool.imap(simular, args_list):
            sim_not_done -= 1
            with open("simulations_log.txt", "a") as log_file:
                log_file.write(f"Simulaciones restantes: {sim_not_done}, Tiempo de simulación: {sim_time/60} minutos\n")


if __name__ == "__main__":
    presiones = np.linspace(10, 50, 5)        # de 10 a 50 Torr
    ds_col = np.linspace(5, 50, 5)            # de 5 a 50 mm
    xs_beam = np.linspace(-4000, 4000, 9)     # de -4cm a 4cm 
    ys_beam = np.linspace(-4000, 4000, 9)    
  
    main(presiones, ds_col, xs_beam, ys_beam)
