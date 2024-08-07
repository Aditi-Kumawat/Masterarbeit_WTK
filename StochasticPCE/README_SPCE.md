## SCPE 
SPCE is a non-intrusive parametric probability based surrogate model. Please check:
 - [Sudret 2014] Sudret, Bruno: Polynomial chaos expansions and stochastic finite element methods. In: Risk and reliability in geotechnical engineering (2014), p. 265–300
 - [Zhu and Sudret 2023] Zhu, Xujia ; Sudret, Bruno: Stochastic polynomial chaos expansions to emulate stochastic simulators. In: International Journal for Uncertainty Quantification 13 (2023), Nr. 2 
  
for detailed implementation of SPCE algorithm.

## Train model and prediction
To train the model, one has to prepare three parts, which are:

1. [Input datasets, X](./InputData/)
2. [Their corresponding realization, Y](./InputData/Realization/)
3. [Configuration file](./config/)

Please check the [example](./Example.ipynb) for detailed implementation.
