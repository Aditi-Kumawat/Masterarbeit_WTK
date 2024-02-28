clear;
clc;

%% Set the number of samples
numSamples = 1000;
numVars = 4;
type_list = ["Gaussian","Gaussian","Gaussian","Gaussian"];

%% DOE
lhsSamples = lhsdesign(numSamples, numVars);
X = zeros(numSamples, numVars);
for i = 1:numVars
    X_i = fns_InvTras_generator(lhsSamples(:, i),type_list(i));
    X(:,i) = X_i;
end

file_name = sprintf('SDOF_AGM_DOE_%d.mat', numSamples);
save(file_name,'X','-mat');

