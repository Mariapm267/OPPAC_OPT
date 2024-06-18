import subprocess
import numpy as np
from multiprocessing import Pool
import os
import time



def simular(args, simulation_executable = '../../OPPAC_Sim/build/exe_scinti'):
    start_time = time.time()  # Registro de tiempo de inicio de la simulación
    n_events, d_col, output_name, pres, x_beam, y_beam = args
    for _ in range(n_events):
        subprocess.run([f"{simulation_executable}", "1", f'{d_col}', output_name, f'{pres}', f'{x_beam}', f'{y_beam}'])
    end_time = time.time()  # Registro de tiempo de finalización de la simulación
    tiempo_simulacion = end_time - start_time  # Cálculo del tiempo transcurrido
    return tiempo_simulacion

def main(presiones, ds_col, xs_beam, ys_beam, num_cores):
    args_list = []
    sim_not_done = 0
    for p in presiones:
        for d in ds_col:
            for x in xs_beam:
                for y in ys_beam:
                    output_name = f"/scratch04/maria.pereira/TFM/Datasets/simu_col{d}_p{p}_x{x}_y{y}"
                    if os.path.exists(output_name):  # Check if the file exists
                        with open(output_name, 'r') as file:
                            lines = file.readlines()
                        if len(lines) < 10000:  # Check if the file has fewer than 10000 lines
                            remaining_events = 10000 - len(lines)
                            args_list.append((remaining_events, d, output_name, p, x, y))
                            sim_not_done += 1
                            print('not completed')
                    else:
                        args_list.append((10000, d, output_name, p, x, y))
                        sim_not_done += 1
                        
    total_simulations = len(presiones)*len(ds_col)*len(xs_beam)*len(ys_beam)

    with open("simulations_log.txt", "w") as log_file:
        log_file.write(f"Numero total de simulaciones: {total_simulations}\n")
        log_file.write(f"Total de simulaciones que quedan: {sim_not_done}\n")

    with Pool(num_cores) as pool:
        for tiempo_simulacion in pool.imap(simular, args_list):
            sim_not_done -= 1
            with open("simulations_log.txt", "a") as log_file:
                log_file.write(f"Simulaciones restantes: {sim_not_done}, Tiempo de simulación: {tiempo_simulacion/60} minutos\n")


if __name__ == "__main__":
    presiones = np.linspace(10, 50, 5)        # de 10 a 50 Torr
    ds_col = np.linspace(5, 50, 5)            # de 5 a 50 mm
    xs_beam = np.linspace(-4000, 4000, 9)     # de -4cm a 4cm 
    ys_beam = np.linspace(-4000, 4000, 9)    

    num_cores = 20  
    main(presiones, ds_col, xs_beam, ys_beam, num_cores)
