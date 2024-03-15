## File naming rules

### SDOF example

1. DOE files 
   - Input: 
     - TEST ID
     - Num Vars
     - Training type: TR / VAL
     - Num of samples
     - Direciton: X / Y / Z
   - Output file name example
     - TEST (ID) _ X _ SBAGM _ V (Num Vars) _ (Train type) _ (Num samples) _ DIR _(Direciton)
     - i.e. TEST0_X_SBAGM_V5_TR_DOE_1000_DIR_Z.mat

2. Realization 
   - Input: 
     - TEST ID
     - Num Vars
     - Training type: TR / VAL
     - Realization type: RAND / FIX + Index
     - Num of samples
     - Direciton: X / Y / Z
   - Output file name example
     - TEST (ID) _ Y _ SBAGM _ V (Num Vars) _ (Train type) _ (Realization type) _  (Num samples) _ DIR _(Direciton)
     - i.e. TEST0_Y_SBAGM_V5_TR_RAND_DOE_1000_DIR_Z.mat
     - i.e. TEST0_Y_SBAGM_V5_TR_FIX200_DOE_1000_DIR_Z.mat

3. Config file
   - Input: 
     - TEST ID
   - Output file name example
     - config_SDOF_ SBAGM_ (ID)
     - i.e. config_SDOF_SBAGM_5.json

