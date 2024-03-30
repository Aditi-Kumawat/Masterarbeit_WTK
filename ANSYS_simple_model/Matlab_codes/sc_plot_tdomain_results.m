
%% Initialization

clear;clc;
% close all
% Define the number of storeys, rooms in x-y-direction

n_str = 1;
n_rx = 1;
n_ry = 1;

% Define the length, width, and height of the building
l_vect=5;
b_vect=3;
h = 3;

% Define the type of foundation as either 'PLATE' or 'FOOTING'
ftyp = 'FOOTING';
% Define the foundation behaviour and analysis type
st_dtls='SSnoVs30Bld_DR_0pt05';
% st_dtls='SSnoVs30Bld_DR1002DR2004';
st_dtls='SSnoVs30BldNoDmp';

% Define the velocity of the excitation
V_s = 450;

% Define the size of the elements
n_esize = 1;

% Calculate the length and width of the footing based on the
% foundation type
if strcmp(ftyp,'PLATE')
    B_f = n_esize/2;
    L_f = n_esize/2;
else
    B_f = 0.75;
    L_f = 0.75;
end

%% Importing Data

ff_fldr = 'GM/SeisSol/data_noVs30_Bld_new';
% Define the file name and path
fil_nm = 'vt_rec_1324.txt';
% funs_plot_properties.import_tim_data(fref_data_folder,filename)
cd ..
% Combine the directory path and file name using the file separator
fil_pth = [ff_fldr filesep fil_nm];

% Import the data
data = readtable(fil_pth);
cd Matlab_codes

% Assign column names
data.Properties.VariableNames={'t_vect', 'data_x', 'data_y', 'data_z'};
rec_vect=[1324 1325 1326];
%%
% Define the name of the folder where the results are stored
rf_fldr = 'tDmn_input';

% Define the base file name for the U center results in ANSYS
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';

% Define the column names in the results table
cols = {'time', 'val'};

% Define the components to extract from the results
cmpt = {'X', 'Y', 'Z'};
n_c = length(cmpt);
figure
for i_c=1:n_c
    subplot(n_c,1,i_c)
    plot(data.t_vect, data{:,i_c+1},'DisplayName',cmpt{i_c},...
        'LineWidth', 1.2)
    xlabel('Time (s)','Interpreter','latex','FontSize',12);
    ylabel('Velocity (m/s)','Interpreter','latex','FontSize',12);
    legend show
    legend('Box','off','Interpreter','latex','FontSize',11)
%     title(sprintf('Component %d', i_c),'Interpreter','latex',...
%         'FontSize',11);
    set(gcf,'Units','inches', 'Position', [18 3 4.5 6],...
        'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
    fil_nm = ['FF_vel_t', '.emf'];
    cd SAVE_FIGS
    % Create the directory (if it doesn't already exist)
    if ~exist(rf_fldr, 'dir')
        mkdir(rf_fldr);
    end

    % Save the figure in the results folder
    saveas(gcf, fullfile(rf_fldr, fil_nm));
    cd ..
    cd ..
    cd Matlab_codes
end
%%


%%

ut_mat=cell(1, n_c);
u_fft=cell(1, n_c);
vt_mat=cell(1, n_c);

for i_str = 0:n_str
    for i_l=1:length(l_vect)
        l=l_vect(i_l);
        for i_b=1:length(b_vect)
            b=b_vect(i_b);
            if b>l
                break
            end
            fldr = fns_plot.get_fldrnm_rec(n_str, n_rx, n_ry,...
                l, b,ftyp, V_s, L_f, B_f,st_dtls);
            filNms = arrayfun(@(x) sprintf(bf_nm, x{1}, i_str, l, b),...
                cmpt, 'UniformOutput', false);

            cd ..
            cd Results_Ansys
            fil_pths = fullfile(rf_fldr, fldr, filNms);
            U_all = cellfun(@(x) readtable(x), fil_pths,...
                'UniformOutput', false);
            cd ..
            cd Matlab_codes

            for i_c = 1:n_c
                U = U_all{i_c};
                U.Properties.VariableNames = cols;
                %                 if i_l == 1
                time = U.time;
                dt=time(3)-time(2);
                ut_mat{i_c} = U.val;
                %%
                vt_mat{i_c}=[0;diff(ut_mat{i_c})]./dt;

            end
        end
    end
    %%
    rec_f=rec_vect(i_str+1);
    fil_nm1 = strcat(sprintf('vt_rec_%d',rec_f), '.txt');
    % funs_plot_properties.import_tim_data(fref_data_folder,filename)
    cd ..
    % Combine the directory path and file name using the file separator
    fil_pth = [ff_fldr filesep fil_nm1];

    % Import the data
    data = readtable(fil_pth);
    cd Matlab_codes

    % Assign column names
    data.Properties.VariableNames = {'t', 'data_x', 'data_y', 'data_z'};
    %%
    sig_mat=vt_mat;
    figure
    for i_s = 1:n_c

        for i_col = 1:size(sig_mat{i_s}, 2)
            sig_vect = sig_mat{i_s}(:, i_col);

            % Plot the time-domain signal
            %         figure
            l=l_vect(i_col);
            b=b_vect(i_col);
            txt = ['SeisSol';'LPM-FEM'];
            %             txt = 'LPM-FEM';
            txt_2 = ['Floor:',num2str(i_str),...
                ',~Component:',num2str(i_s)];
            subplot(n_c, size(sig_mat{i_s}, 2),...
                (i_s-1)*size(sig_mat{i_s}, 2) + i_col);
            plot(data.t, data{:,i_s+1}, 'LineWidth', 1.5)
            hold on
            plot(time-0.005, real(sig_vect),'LineStyle',':','LineWidth', 1.5);
            xlabel('Time (s)','Interpreter','latex','FontSize',12);
            ylabel('Velocity (m/s)','Interpreter','latex','FontSize',12);
            if i_s==1 && i_str==2
                ylim([-5e-4, 5e-4])
            elseif i_s==2 && i_str==2
                ylim([-2e-3, 2e-3])
            elseif i_s==3 && i_str==2
                ylim([-5.5e-3, 5.5e-3])
            end
            %             title(txt_2,'Interpreter','latex','FontSize',11);
            legend(txt);
            legend('Box','off','Interpreter','latex','FontSize',11)
            set(gcf,'Units','inches', 'Position', [18 3 4.5 6],...
                'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
        end
    end
    fil_nm = ['Tdomain_vel_t_','_Floor_', num2str(i_str),...
        '_ftyp_', ftyp,'_Vs_', num2str(V_s),...
        '_Lf_', num2str(L_f), '_Bf_', num2str(B_f), '.emf'];
    cd SAVE_FIGS
    % Create the directory (if it doesn't already exist)
    if ~exist(rf_fldr, 'dir')
        mkdir(rf_fldr);
    end

    % Save the figure in the results folder
    saveas(gcf, fullfile(rf_fldr, fil_nm));
    cd ..
    cd ..
    cd Matlab_codes
end
