classdef fns_VsVary
    methods (Static)
        %%
        function [f_vect,TFamp_mat,TFcmplx_mat]=...
                get_TFUniBldVsVary(n_str, n_rx, n_ry,l, b,...
                ftyp, Vs_vect, L_f, B_f,bf_nm,i_str,cmpt,n_c,r_fldr,cols)

          
            % Loop over vs
            for i_vs = 1:size(Vs_vect, 1)
                V_s=Vs_vect(i_vs);
                % Get the folder name for the specific configuration
                fldr = fns_plot.get_fldrnm(n_str,...
                    n_rx, n_ry, l, b, ftyp, V_s, L_f, B_f);
                fl_nm1 = arrayfun(@(x) sprintf(bf_nm, x{1}, i_str,...
                    V_s),cmpt, 'UniformOutput', false);
                cd ..
                cd Results_Ansys
                fil_pth = fullfile(r_fldr, fldr, fl_nm1);
                U_all = cellfun(@(x) readtable(x), fil_pth,...
                    'UniformOutput', false);
                cd ..
                cd Matlab_codes

                for i_component = 1:n_c
                    U = U_all{i_component};
                    U.Properties.VariableNames = cols;
                    if i_vs == 1
                        f_vect = U.Freq;
                        TFamp_mat{i_component} = U.AMPL;
                        TFr_mat{i_component} = U.REAL;
                        TFim_mat{i_component} = U.IMAG;
                        TFcmplx_mat{i_component} = U.REAL+1i.*U.IMAG;
                    else
                        TFamp_mat{i_component}(:, i_vs) = U.AMPL;
                        TFr_mat{i_component}(:, i_vs)  = U.REAL;
                        TFim_mat{i_component}(:, i_vs)  = U.IMAG;
                        TFcmplx_mat{i_component}(:, i_vs)  = ...
                            U.REAL+1i.*U.IMAG;
                    end
                end
            end
        end
    end
end