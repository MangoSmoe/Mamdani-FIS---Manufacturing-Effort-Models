# Mandani Fuzzy Inference Systems (FIS)

Institute of Lightweight Structures
TU Munich
 
Last Update: 12.12.2018

## Author:
Dr Markus Edwin Schatz (schatz@llb.mw.tum.de)



## Description:
This library contains basic Mandani fuzzy inference system classes. Moreover, a manufacturing effort model for predicting effort and costs of braided structures is provided. For more details consult the following papers:

'''
M. Schatz, A. Hermanutz, H. Baier: **Multi-criteria optimization of an aircraft propeller considering manufacturing**, Structural and Multidisciplinary Optimization Vol. 55, Issue 3, pp 899–911, https://doi.org/10.1007/s00158-016-1541-z
'''

## How to run:
- See each class' comments
- Simply run "BraidME.py" and chose test case via variable TestCase
- The following test cases are defined via variable TestCase:
   - **TestX**: Just evaluate effort
   - **TestXSens**: Compute effort and sensitivities
   - **PlotSurfaces**: Plot all relevant response surfaces
   - **PlotRules**: Plot all implemented rules
   - **Costs**: Compute costs of given design
   - **MEInpDict**: Generate input dictionary
   - **MEInpDictSens**: Compute input and sensitivities
- Or directly consult author by mail

## Implemented features
- Generates Mamdani FIS
- Plots all rules of Mamdani FIS
- Plots all relevant response surfaces of FIS
- Compute manufacturing effort for braiding
- Provides reasoning and advice for a given scenerio
- Computes production costs for braiding
- Computes sensitivities of effort model


## Planned features:
- Further manufacturing effort models

