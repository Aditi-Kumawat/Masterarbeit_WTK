
%% Initialization

clear;clc;
close all

% Define the number of storeys, rooms in x-direction, and y-direction
n_str = 2;
n_rx = 2;
n_ry = 3;

% Define the length, width, and height of the building
l_vect=5;
b_vect=3;
h = 3;

% Define the type of foundation as either 'PLATE' or 'FOOTING'
ftyp = 'PLATE';
% Define the foundation behaviour and analysis type
rec=936;
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
%%
ifft_dataFldr='SAVE_DATA/Results_Freq_Indepn_TF';
%%
% Define the name of the folder where the results are stored
rf_fldr = 'Results_t_domain/SeisSol/data_Vs30_Bld';

% Define the base file name for the displacement center results in ANSYS
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';
% Define the column names in the results table
cols = {'time', 'val'};
% Define the components to extract from the results
cts = {'X', 'Y', 'Z'};
n_c = length(cts);
%%
ut_mat=cell(1, n_c);
vt_mat=cell(1, n_c);

for i_str = 0:n_str
    fil_nm1 = strcat(sprintf('vt_SeisSol_ifft_%d',i_str), '.txt');
    % funs_plot_properties.import_tim_data(fref_data_folder,filename)

    % Combine the directory path and file name using the file separator
    fil_pth = [ifft_dataFldr filesep fil_nm1];
    % Import the data
    data_ifft = readtable(fil_pth);


    data_ifft.Properties.VariableNames = {'t_if', 'v_x', 'v_y', 'v_z'};
    for i_l=1:length(l_vect)
        l=l_vect(i_l);
        for i_b=1:length(b_vect)
            b=b_vect(i_b);
            if b>l
                break
            end

            fldr = fns_plot.get_fldrnm_rec(n_str, n_rx, n_ry, l, b,...
                ftyp, V_s, L_f, B_f,rec);
            filNms = arrayfun(@(x) sprintf(bf_nm, x{1}, i_str, l, b),...
                cts, 'UniformOutput', false);

            cd ..
            fil_pths = fullfile(rf_fldr, fldr, filNms);
            U_all = cellfun(@(x) readtable(x), fil_pths,...
                'UniformOutput', false);
            cd Matlab_codes
            for i_component = 1:n_c
                U = U_all{i_component};
                U.Properties.VariableNames = cols;
                %                 if i_l == 1
                time = U.time;
                dt=time(3)-time(2);
                ut_mat{i_component} = U.val;
                %%
                vt_mat{i_component}=[0;diff(ut_mat{i_component})]./dt;
            end
        end
    end
    %%
    sig_mat=vt_mat;
    t_if=data_ifft.t_if;
    figure
    for i_s = 1:n_c
        dat_if=data_ifft{:,i_s+1};
        for i_col = 1:size(sig_mat{i_s}, 2)
            sig_vect = sig_mat{i_s}(:, i_col);

            % Plot the time-domain signal
            %         figure
            l=l_vect(i_col);
            b=b_vect(i_col);
            txt = {'IFFT';'Transient'};
            %             txt = 'LPM-FEM';
            txt_2 = ['Floor:',num2str(i_str),',~Component:',num2str(i_s)];
            subplot(n_c, size(sig_mat{i_s}, 2),...
                (i_s-1)*size(sig_mat{i_s}, 2) + i_col);
            plot(t_if, dat_if, 'LineWidth', 1.2)
            hold on
            plot(time, real(sig_vect),'LineWidth', 1.2);
            xlabel('Time (s)','Interpreter','latex','FontSize',10);
            ylabel('Velocity (m/s)','Interpreter','latex','FontSize',10);
            title(txt_2,'Interpreter','latex','FontSize',11);
            legend(txt);
            legend('Box','off','Interpreter','latex','FontSize',11)
            set(gcf,'Units','inches', 'Position', [18 3 4.5 6],...
                'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
        end
    end
    filename = ['Cmpr_velt_ifft_trans_','_Floor_', num2str(i_str),...
        '_ftyp_', ftyp,'_Vs_', num2str(V_s),...
        '_Lf_', num2str(L_f), '_Bf_', num2str(B_f), '.png'];
    cd SAVE_FIGS
    if ~exist(rf_fldr, 'dir')
        mkdir(rf_fldr);
    end
    saveas(gcf, fullfile(rf_fldr, filename));
    cd ..
    cd ..
    cd Matlab_codes
end
