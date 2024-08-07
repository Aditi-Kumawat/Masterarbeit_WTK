## SBGMG

SBGMG can generate the artificial ground motion from seismic parameter such as moment magnitude, rupture distacance and so on.

For the implementation, please check:

Hu model: 
- [Chen et al 2022] Chen, Bo ; Sun, Guangjun ; Li, Hongjing: Power spectral models of stationary earthquake-induced ground motion process considering site characteristics. In: Emergency Management Science and Technology 2 (2022), Nr. 1, p. 1–12

Time modulating function:

- [Rezaeian 2010] Rezaeian, Sanaz: Stochastic modeling and simulation of ground motions for performance-based earthquake engineering. University of California, Berkeley, 2010
  
- [Rezaeian and Der Kiureghian 2010] Rezaeian, Sanaz ; Der Kiureghian, Armen: Simulation of synthetic ground motions for specified earthquake and site characteristics. In: Earthquake Engineering & Structural Dynamics 39 (2010), Nr. 10, p. 1155–1180

Linear mixed effect model:

- [Rezaeian 2010] Rezaeian, Sanaz: Stochastic modeling and simulation of ground motions for performance-based earthquake engineering. University of California, Berkeley, 2010
- [The MathWorks Inc. 2022] The MathWorks Inc.: linear mixed-effects model, MATLAB version: 9.13.0 (R2022b). (2022). – URL https://www.mathworks.com/help/stats/linearmixedmodel.html


Baseline correction:

- [Pan et al 2016] Pan, Chao ; Zhang, Ruifu ; Luo, Hao ; Shen, Hua: Baseline correction of vibration acceleration signals with inconsistent initial velocity and displacement. In: Advances in Mechanical Engineering 8 (2016), Nr. 10, p. 1687814016675534

Spectral response analysis:
- [Mostafa Tazarv 2024] Mostafa Tazarv: Elastic Response Spectrum. (2024).– URL https://www.mathworks.com/matlabcentral/fileexchange/31254-elastic-response-spectrum


### Requirement
To run the SBGMG, it requires the database (not provide), the [fitted model parameters and further information from station](./MAT_FILES/) for training.



### Example
Please check [Example_SpectralRespAnalysis.m](./Example_SpectralRespAnalysis.m) for the implementation of SBGMG and the validation using Spectral Response Analysis. 

