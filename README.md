# multiscale_dev_dynamics

## What is this repo for?
- Analysing whole brain single cell calcium imaging data from larval zebrafish over development (3,4,5,6,7,8 dpf)
- Link microscale properties with global dynamics
- Using reservoir pool networks to model synaptic connections in data via FORCE learning
- Simulate network perturbations using spiking network models 

##What does this repo contain?
Modules contain functions for multiscale dynamics analyses and modelling
Accompanying ipynotebooks demonstrate how to use the modules

## Modules
'metastability.py' - functions for calculating the number of metastable states and state transition dynamics

## Notebooks
'dev_dynamics.ipynb' - quantifying global dynamics over development

'FORCE_model.ipynb' - using tension FORCE learning to infer the effective connections driving observed whole brain dynamics

'av_dev.ipynb' - quantifying the propagation of avalanches over development
