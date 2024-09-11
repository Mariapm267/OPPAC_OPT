
This repository is focused on the optimization of a detector (Parallel-Plate Avalanche Counter with Optical Readout) employing differentiable programming and machine learning techniques.

# Detector description:


The Optical Parallel-Plate Avalanche Counter is a detector designed for heavy ion tracking and imaging. It consists of 2 parallel plates separated by a 3 mm gap with a high electroluminiscense gas mixture (CF4) inside. When a charged particle crosses the PPAC, it ionizes the medium and avalanches are produced. Then, secondary scintillation processes produce photons that are detected on the walls of the PPAC by small collimated photosensors (SiPMs). The photon distributions detected on each wall are used to reconstruct the position of the impining particle with high precission.

There are two parameters that affect the detector resolution and that we want to optimize:
 - Pressure of scintillating gas ($p$): in general, higher pressure means better statistics in the photon distributions.
 - Collimator Length ($L$): the larger the collimator, the better the distribution std but the lower the statistics. 


Above, a simulation of the detector is shown for an impining alpha particle:

<img width="664" alt="Screenshot 2024-03-13 at 22 33 35" src="https://github.com/user-attachments/assets/a1d7cf35-dce6-4148-a2ab-1ec5cf47d799">

Reference: [2]




# This repository...


...contains several tools for the reconstruction of the position of the impining particle, techniques to develop and train a Neural Network that will predict the detector output from the detector configuration and the beam position ($x, y, p, L$) and the tools needed to optimize the detector resolution using differentiable programming and automatic differentiation.

This is a derivative work from: https://github.com/GilesStrong/tomopt, TomOpt: Differential Muon Tomography Optimisation


## How does it work:

 -   Simulation. First, a grid of parameter combinations is simulated with Geant4. The output consists of the number of photons detected  in each SiPM.
  
 -   Preprocessing: at this point we have two options:
   
     - From simulated events, reconstruct the position (with this option, the NN will predict the reco position directly).
     - From simulated events, reconstruct the photon distribution (N, $\mu$, $\sigma$) in each wall. NOT FULLY IMPLEMENTED YET
   
 -   Split data: examples/run_split_data.py will split data into hyperparameter tuning datasets, training datasets and an evaluation dataset.
  
 -   Hyperparameter tuning: done with examples/run_optuna.py
    
 -   Training: In this step the Neural Network is trained to predict the detector output (either the photon distributions or the reconstructed position). This is done with examples/run_nn.py and the model is saved as a .pt to be used later in the differentiable pipeline.
    
 -   Optimization: in the optimization, an objective function that will be minimize in the detector optimization is defined (reconstruction error). The detector parameters are established as 'free' parameters that can evolve according to the gradient of the objective function with respect to these parameters. Using autodiff, we construct an optimization loop that will search for the minimum of this function. This can be done using examples/run_opt.py
   

# References: 

[1] G. C. Strong, M. Lagrange, A. Orio, A. Bordignon, F. Bury, T. Dorigo, A. Gi-
ammanco, M. Heikal, J. Kieseler, M. Lamparth, P. Martínez Ruíz del Árbol, F.
Nardi, P. Vischia, and H. Zaraket, “TomOpt: Differential optimisation for task- and
constraint- aware design of particle detectors in the context of muon tomography”,
ArXiv:2309.14027, 2023.

[2] M. Cortesi, Y. Ayyad, and J. Yurkon, “Development of a parallel-plate avalanche
counter with optical readout (O-PPAC)”, Journal of Instrumentation, vol. 13, no.
10, pp. P10006-P10006, 2018
