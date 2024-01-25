clear;
clc;

% Set the number of samples
numSamples = 500;

% Specify the parameter ranges for each variable
massRange = [0,1];        % Example: Mass follows a normal distribution with mean 0 and standard deviation 1
dampingRange = [0,1];     % Example: Damping follows a normal distribution with mean 0 and standard deviation 1
stiffnessRange = [-1,1];  % Example: Stiffness follows a uniform distribution between -1 and 1

% Generate Latin hypercube samples
lhsSamples = lhsdesign(numSamples, 3);

% Map the samples to the specified parameter ranges
massSamples = norminv(lhsSamples(:, 1), massRange(1), massRange(2));          % Inverse of cumulative distribution function for normal distribution
dampingSamples = norminv(lhsSamples(:, 2), dampingRange(1), dampingRange(2));     % Inverse of cumulative distribution function for normal distribution
stiffnessSamples = lhsSamples(:, 3) * (stiffnessRange(2) - stiffnessRange(1)) + stiffnessRange(1);  % Map to uniform distribution

X = [massSamples,dampingSamples,stiffnessSamples];
file_name = sprintf('samples_%d.mat', numSamples);
save(file_name,'X','-mat');