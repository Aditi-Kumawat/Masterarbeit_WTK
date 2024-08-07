clear;
clc;

%% Set the number of samples
TEST_CASE = 2;
dir = 'Z';
numSamples = 1000;
numVars = 6;

Learning_type = 'TR';
%Learning_type = 'VAL';

DR_ANSYS = 6;
E_ANSYS = 5;

type_list = ["Gaussian","Uniform","Uniform","Gaussian","Gaussian","Gaussian"];

%% DOE
lhsSamples = lhsdesign(numSamples, numVars);
X = zeros(numSamples, numVars);

for i = 1:numVars
    X_i = fns_InvTras_generator(lhsSamples(:, i),type_list(i));
    X(:,i) = X_i;
    
end

output_file_name = sprintf('TEST%d_X_SBAGM_V%d_%s_DOE_%d_DIR_%s.mat',TEST_CASE,numVars,Learning_type,numSamples,dir);

% ANSYS_file name
DR_file_name = sprintf('ANSYS_DR_TEST%d_X_SBAGM_V%d_%s_DOE_%d.txt',TEST_CASE,numVars,Learning_type,numSamples);
E_file_name = sprintf('ANSYS_E_TEST%d_X_SBAGM_V%d_%s_DOE_%d.txt',TEST_CASE,numVars,Learning_type,numSamples);


X_DR_ANSYS= exp(-3.2+0.1*X(:,DR_ANSYS));
fid = fopen(DR_file_name, 'w');
fprintf(fid, ' %14.7e \n', X_DR_ANSYS');
fclose(fid);
%type 'DR_file_name'


E_mean = 30e9;
X_E_ANSYS= E_mean + E_mean*0.05*X(:,E_ANSYS);
fid = fopen(E_file_name, 'w');
fprintf(fid, ' %14.7e \n', X_E_ANSYS');
fclose(fid);
%type 'E_file_name'

save(output_file_name,'X','-mat');

