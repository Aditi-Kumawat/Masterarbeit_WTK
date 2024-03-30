clear;
clc;

%% Set the number of samples
TEST_CASE = 2;
dir = 'Z';
numSamples = 1000;
numVars = 6;

Learning_type = 'TR';
%Learning_type = 'VAL';

output_file_name = sprintf('TEST%d_X_SBAGM_V%d_%s_DOE_%d_DIR_%s.mat',TEST_CASE,numVars,Learning_type,numSamples,dir);

X = load(output_file_name);
X_all = X;
SplitSamples = 500;

% Split 
X = X_all.X(2:501,:);
Learning_type = 'TR';
output_file_name = sprintf('TEST%d_X_SBAGM_V%d_%s_DOE_%d_DIR_%s_SPLIT.mat',TEST_CASE,numVars,Learning_type,SplitSamples,dir);
save(output_file_name,'X','-mat');


X = X_all.X(501:1000,:);
Learning_type = 'VAL';
output_file_name = sprintf('TEST%d_X_SBAGM_V%d_%s_DOE_%d_DIR_%s_SPLIT.mat',TEST_CASE,numVars,Learning_type,SplitSamples,dir);
save(output_file_name,'X','-mat');


