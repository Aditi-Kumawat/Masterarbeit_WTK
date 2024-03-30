%% Initialization
clear;clc;close all;

% Define the number of storeys, rooms in x- y-direction
n_str = 3;
n_rx = 2;
n_ry = 3;

% Define the length, width, and height of the building
l_vect=[5];
b_vect=[5];
% l_vect=[3 5 7];
% b_vect=[3 5 7];
h = 3;

% Define the type of foundation as either 'PLATE' or 'FOOTING'
ftyp = 'PLATE';

% Define the velocity of the excitation
V_s = 450;

% Define the size of the elements
n_esize = 0.5;

% Calculate the length and width of the footing based on the
% foundation type
if strcmp(ftyp,'PLATE')
    B_f = n_esize/2;
    L_f = n_esize/2;
else
    B_f = 0.75;
    L_f = 0.75;
end


