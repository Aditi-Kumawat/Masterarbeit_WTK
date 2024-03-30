%%
clear;clc;
close all;
% Define the number of storeys, rooms in x-y-direction
n_str = 1;
n_rx = 1;
n_ry = 1;
% Define the length, width, and height of the building
l = 4;
b = 4;
h = 3;
% Define the type of foundation as either 'PLATE' or 'FOOTING'
ftyp = 'FOOTING';
V_s =40;
rho_s=1260;
% Define the size of the elements
n_esize = 0.25;
if strcmp(ftyp,'PLATE')
    B_f = n_esize/2;
    L_f = n_esize/2;
else
    B_f = 0.75;
    L_f = 0.75;
end
t_f=0.2;
n_f=4;
vol_fndn=4*L_f*B_f*t_f*n_f;
rho_bld=2500;

vS_vect =[40 155 270 385 500];
rhos_vect=[1260,1570,1880,2190,2500];
%% Importing Transfer Function
rf_fldr = 'UnitBld_VsVarySSIcplng';
bf_nm = 'Disp_Center_%s_%d_Vs%d';
rf_fldr1 = 'UnitBld_VsVary';
bf_nm1 = 'Disp_Center_%s_%d_Vs%d';
cols1 = {'Freq', 'AMPL','PHASE','REAL','IMAG'};
cmpt = {'X', 'Y', 'Z'};
n_c = length(cmpt);

%%
%floor 0 values are all 1 as the load is applied at the footing
i_str1 = 1;
i_str0 = 0;
V_s1=vS_vect(1);
% [f_out,Uamp_mat,Ucpmlx_mat]=fns_imprtdata.get_TFvSvary(...
%     n_str, n_rx, n_ry,...
%     l, b, ftyp, V_s1, L_f, B_f,...
%     bf_nm,i_str1,cmpt,n_c,rf_fldr,cols1);
for i_v=1:length(vS_vect)
    V_s=vS_vect(i_v);
    [f_out,Uamp_mat,Ucpmlx_mat]=fns_imprtdata.get_TFvSvary(...
        n_str, n_rx, n_ry,...
        l, b, ftyp, V_s, L_f, B_f,...
        bf_nm,i_str1,cmpt,n_c,rf_fldr,cols1);
    [f_out,Uamp_mat_onlyfooting,Ucpmlx_mat_onlyfooting]=...
        fns_imprtdata.get_TFvSvary(...
        n_str, n_rx, n_ry,...
        l, b, ftyp, V_s, L_f, B_f,...
        bf_nm1,i_str0,cmpt,n_c,rf_fldr1,cols1);
    Ucpmlx_footing(i_v,:)=Ucpmlx_mat_onlyfooting;
    Ucpmlx_bld(i_v,:)=Ucpmlx_mat;

end

%%
f_vect1=f_out;
u_Rmat=zeros(length(vS_vect),length(f_vect1));
for i_c=1:3

    for i_v=1:length(vS_vect)
        Ucpmlx_bldmat=Ucpmlx_bld(i_v,:);
        U_bld=Ucpmlx_bldmat{i_c};
        u1_mat=Ucpmlx_footing(i_v,:);
        u1_vect=u1_mat{i_c};
        u_R=(U_bld)./(u1_vect);
        u_Rmat(i_v,:)=abs(u_R);
    end

    figure
    hold on
    leg_vect = cell(1, length(vS_vect)-1);
    for i_v = 1:length(vS_vect)
        plot(f_vect1, 20*log10(u_Rmat(i_v,:)), 'LineWidth', 1.5)
        leg_vect{i_v} = sprintf('$V_s =$ %d m/s', vS_vect(i_v));
    end

    legend(leg_vect, 'Box', 'off', 'Interpreter', 'latex',...
        'FontSize', 12)
    xlabel({'Frequency (Hz)'}, 'FontSize', 12,...
        'Interpreter', 'latex')
    ylabel('$20log_{10}(u_{b}/u_{f}$)~(db)', 'FontSize', 12,...
        'Interpreter', 'latex')
    txt = [cmpt{i_c},'~--~direction',];
    title(txt,'Interpreter','latex','FontSize',11);
    set(gca, 'XTickLabelMode', 'auto');
    set(gca, 'YTickLabelMode', 'auto');
    set(gca,'FontSize',10, 'Box', 'on','LineWidth',1,...
        'TickLabelInterpreter','latex',...
        'TickLength',[0.01,0.01]);
    box on
    set(gcf, 'Units', 'inches', 'Position', [30 4 5 2.5],...
        'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
    filename = ['URatio_UnitBld_abs_', num2str(i_c),...
        '_l', num2str(l), '_by_b', num2str(b),...
        '_ftyp_', ftyp, '_Vsmin_', num2str(V_s),...
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



