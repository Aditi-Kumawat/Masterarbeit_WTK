clear;clc;
% close all
n_str = 2;
n_rx = 2;
n_ry = 3;

l_vect=[3 4 5 6];
b_vect=[3 4 5 6];
h = 3;

ftyp = 'PLATE';
% ftyp = 'FOOTING';
V_s = 450;
n_esize = 0.25;
if strcmp(ftyp,'PLATE')
    B_f = n_esize/2;
    L_f = n_esize/2;
else
    B_f = 0.75;
    L_f = 0.75;
end
%% Importing Transfer Function
% rf_fldr = 'Results_Freq_Indepn_TF';
rf_fldr = 'MultiUnitBld_GeomVary';
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';
cols = {'Freq', 'AMPL','PHASE','REAL','IMAG'};
cmpt = {'X', 'Y', 'Z'};
n_c = length(cmpt);
for i_str = 0:n_str
    for i_l=1:length(l_vect)
        l=l_vect(i_l);
        b=b_vect(i_l);
        fldr = fns_plot.get_fldrnm(n_str, n_rx, n_ry,...
            l, b,ftyp, V_s, L_f, B_f);
        fil_nm = arrayfun(@(x) sprintf(bf_nm, x{1}, i_str,...
            l, b),cmpt, 'UniformOutput', false);
        cd ..
        cd Results_Ansys
        fil_pth = fullfile(rf_fldr, fldr, fil_nm);
        U_all = cellfun(@(x) readtable(x), fil_pth,...
            'UniformOutput', false);
        cd ..
        cd Matlab_codes
        for i_c = 1:n_c
            Uc = U_all{i_c};
            Uc.Properties.VariableNames = cols;
            if i_l == 1
                f = Uc.Freq;
                Uamp_mat{i_c} = Uc.AMPL;
            else
                Uamp_mat{i_c}(:, i_l)= Uc.AMPL;
            end

        end
    end
    %% Plotting Transfer Function
    for i_c = 1:n_c
        xl1=5;
        if i_c==3
            xl1=40; 
        end
        fns_plot.plt_TF(f,Uamp_mat{i_c},i_str, n_rx,...
            n_ry, l_vect, b_vect, ftyp,...
            V_s, L_f, B_f,i_c, cmpt,rf_fldr,xl1);
    end
end