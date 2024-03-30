
%% Initialization
clear;clc;close all;

% Define the number of storeys, rooms in x-,y-direction
n_str = 2;
n_rx = 2;
n_ry = 3;

h = 3;
l_vect=5;
b_vect=5;
% Define the type of foundation as either 'PLATE' or 'FOOTING'
ftyp = 'PLATE';
% Define the velocity of the excitation
V_s = 450;

% Define the size of the elements
n_esize = 0.5;
%% Importing Transfer Function


% Define the name of the folder where the results are stored
rf_fldr_wall = 'Bld_with_Walls';
rf_fldr = 'MultiUnitBld_GeomVary';
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';
if strcmp(ftyp,'PLATE')
    B_f = n_esize/2;
    L_f = n_esize/2;
else
B_f = 0.75;
L_f = 0.75;
end
% Define the column names in the results table
cols1 = {'Freq', 'AMPL','PHASE','REAL','IMAG'};

% Define the components to extract from the results
cmpt = {'X', 'Y', 'Z'};
%%
n_c = length(cmpt);
vel_abs_mat_Z_cell = cell(1, n_str+1);
fref_cmplx_mat_1=cell(1, n_c);
DISP_cmplx_mat=cell(1, n_c);
VEL_abs_mat=cell(1, n_c);
fref_vel_amp_mat_1=cell(1, n_c);

for i_str = 0:n_str
    [~,TFamp_wl,TFcpmlx_wl]=fns_imprtdata.get_TF(n_str, n_rx, n_ry,...
        l_vect, b_vect, ftyp, V_s, L_f, B_f,...
        bf_nm,i_str,cmpt,n_c,rf_fldr_wall,cols1);
    [f_vect,TFamp,TFcpmlx]=fns_imprtdata.get_TF(n_str, n_rx, n_ry,...
        l_vect, b_vect, ftyp, V_s, L_f, B_f,...
        bf_nm,i_str,cmpt,n_c,rf_fldr,cols1);
    for i_c = 1:n_c
        %% Plotting Transfer Function
        %         fns_plot.plt_TF(f_off,TFamp{i_c}, TFamp_off{i_c},...
        %                     i_str, n_rx, n_ry, l_vect, b_vect, ftyp,...
        %                     V_s, L_f, B_f, i_c, cmpt);
        ha_col = @colors;
        lStyl = {'-', ':', ':', '-.'};
        lcol = {ha_col('boston university red'), ha_col('cadmium green'),...
                ha_col('black'),ha_col('denim'),...
                ha_col('dark goldenrod')};
        ustr_vect={'$u_x$~(m)','$u_y$~(m)','$u_z$~(m)'};
        figure
        for i_lb=1:length(l_vect)
            l=l_vect(i_lb);
            b=b_vect(i_lb);
            % txt_1 = ['Floor~size:~',num2str(l),'m~x',...
            % num2str(b),'m'];
            % txt_2 = ['Floor~size:~',num2str(l_off),'m~x',...
            % num2str(b_off),'m,~','Offset:~2m'];
            txt_1='frame~without~walls';
            txt_2='frame~with~walls';
            if i_c==1
                i_col=1;
            elseif i_c==2
                i_col=2;
            else
                i_col=4;
            end
            hold on
            plot(f_vect,TFamp{i_c}(:,i_lb),...
                'linestyle',lStyl{mod(i_lb-1,numel(lStyl))+1},...
                'DisplayName',txt_1,'LineWidth',1.5,...
                'Color',lcol{mod(i_lb-1,numel(lcol))+i_col})
            hold on
            plot(f_vect,TFamp_wl{i_c}(:,i_lb),...
                'linestyle',lStyl{mod(i_lb-1,numel(lStyl))+2},...
                'DisplayName',txt_2,'LineWidth',1.5,...
                'Color',lcol{mod(i_lb-1,numel(lcol))+3})
        end
%         xlim([0,40])
        %text_Str_TF=['Floor=~',num2str(i_str),...
        % ',~Transfer~function'];
        %title(text_Str_TF,'Interpreter','latex','FontSize',8)

        fns_plot.setPltProps(ustr_vect,i_c);

        filename = ['TFcmpr_walls',cmpt{i_c},...
            num2str(i_str),'_n_rooms_X_', num2str(n_rx),...
            '_n_rooms_Y_', num2str(n_ry),...
            '_ftyp_', ftyp,'_Vs_', num2str(V_s),...
            '_Lf_', num2str(L_f), '_Bf_', num2str(B_f),'.pdf'];

        cd SAVE_FIGS
        if ~exist(rf_fldr_wall, 'dir')
            mkdir(rf_fldr_wall);
        end
        saveas(gcf, fullfile(rf_fldr_wall, filename));
        cd ..
        cd ..
        cd Matlab_codes
    end
end
