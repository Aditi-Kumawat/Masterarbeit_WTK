## Building model

This is the building model of Geotheraml power plant in Taufkirchen, Munich.
The model is built using ANSYS APDL. Since the SSI part is modeled by MATRIX27 element, therefore please check whether your ANSYS license include this element or not first before running it.   

### Description

1. Generate the freqeuncy response functions (FRF) at location (8.75, 4.8, Z) where Z is the height of each floor.
2. The random Young's modulus and damping ratio are imported from [this](./fn_pre/RANDOM_INPUT/) folder 
3. For the thesis, it will generate 1000 realization and store in [subfoler](./Results_Ansys/) called DataFromServer folder. However, the results are not uploaded here.
4. One simulation might take appr. 9 mins. 

### Running the simulation:

1. To select which running mode, please open ```main_TK_VaryDamping.mac```. Then finding the ```DRtype``` and select the one you want. If ```DRtype = 'vary'```, it will run thought all the random Young's modulus and damping ratio. 
2. Open Mechanical APDL (my version is 2021 R2), and type the comment ``` /INPUT,'main_TK_VaryDamping','mac'```



