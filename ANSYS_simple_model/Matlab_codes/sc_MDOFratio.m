%%
clear;clc;
close all;
% Define the number of storeys, rooms in x-y-direction
n_str = 1;
n_rx = 1;
n_ry = 1;
% Define the length, width, and height of the building
l = 5;
b = 5;
h = 3;
% Define the type of foundation as either 'PLATE' or 'FOOTING'
ftyp = 'FOOTING';
V_s =40;
rho_s=500;
% Define the size of the elements
n_esize = 0.25;
if strcmp(ftyp,'PLATE')
    B_f = n_esize/2;
    L_f = n_esize/2;
else
    B_f = 0.75;
    L_f = 0.75;
end

vS_vect =linspace(100,1500,100);
rhos_vect=linspace(1000,2400,100);
%% XY
l_col=0.3;
b_col=0.3;
l_beam=0.3;
b_beam=0.3;
n_col=4;
n_beamx=2;
n_beamy=2;
t_slb=0.2;
t_f=0.2;
n_f=4;
vol_bm=l_beam*b_beam*(l*n_beamx+b*n_beamy)*n_str;
vol_col=l_col*b_col*h*n_col*n_str;
vol_slb=t_slb*l*b*n_str;
vol_fndn=4*L_f*B_f*t_f*n_f;
rho_bld=2500;
m_bldXY=(vol_bm+vol_col+vol_slb+vol_fndn)*rho_bld;

I_col =(b_col*l_col^3)/12;
E_col=30e9;
k_bldXY=12*E_col*I_col/h^3;
w0_xy=sqrt(k_bldXY/m_bldXY);
f0_xy=w0_xy/2/pi;
dr_bldXY=0.05;
c_bldXY=dr_bldXY*(2*m_bldXY*w0_xy);
%% slab
l_slb=l;
b_slb=b;
m_slb=(vol_bm+vol_slb)*rho_bld;
% m_slb=m_bldXY;
% f_0_slab=70.5;
% a_slab=5.2;
% b_slab=3.9;
gama_S=l_slb/b_slb;
E_slb=30e9;     %Elastic Modulus
beta1=1.57*(5.14+3.13*gama_S^2+5.14*gama_S^4)^0.5;
nu_slb=0.06;    % from Sanayei verical vib models
F0=(E_slb*t_slb^3*l_slb*b_slb)/(12*(1-nu_slb^2)*m_slb);
f0_slb=(beta1/l_slb^2)*sqrt(F0);
w0_slb=2*pi*f0_slb;
k_slb=m_slb*w0_slb^2;
dr_slb=0.05;
c_slb=dr_slb*(2*m_slb*w0_slb);
%%
k_ratio=1; %Ratio of B/C
BLratio_vect=[1 2 4];
BLratio=BLratio_vect(k_ratio);
mfXY=0.5*vol_fndn*rho_bld;
nu=1/3;

G_s=rho_s.*V_s.^2;
KY=((2*G_s*L_f)/(2-nu))*(2+2.5*(BLratio^0.85));
cY=((L_f./V_s).*0.58.*KY);
KZ=((2*G_s*L_f)/(1-nu))*(0.73+1.54*(BLratio^0.75));
cZ=((L_f./V_s).*0.85.*KZ);

G_s_vect=rhos_vect.*vS_vect.^2;
KY_vect=((2*G_s_vect*L_f)/(2-nu))*(2+2.5*(BLratio^0.85));
cY_vect=((L_f./vS_vect).*0.58.*KY_vect);
KZ_vect=((2*G_s_vect*L_f)/(1-nu))*(0.73+1.54*(BLratio^0.75));
cZ_vect=((L_f./vS_vect).*0.85.*KZ_vect);
%%
rf_fldr='EquiMDOF';
f_vect1=5:5:50;
for i_c=2:3
    if i_c==2
        KF_vect=KY_vect;
        cF_vect=cY_vect;
        KF=KY;
        cF=cY;
        k_bld=k_bldXY;
        c_bld=c_bldXY;
        m_bld=m_bldXY;
        mf=mfXY;
    elseif i_c==3
        KF_vect=KZ_vect;
        cF_vect=cZ_vect;
        KF=KZ;
        cF=cZ;
        k_bld=k_slb;
        c_bld=c_slb;
        m_bld=m_slb;
        mf=mfXY;
    end

    for i_o=1:length(f_vect1)
        omg=2*pi*f_vect1(i_o);
        K11=-mf*omg.^2+(KF+k_bld)+1i*omg.*(cF+c_bld);
        K12=-k_bld-1i*omg.*c_bld;
        K21=K12;
        K22=-m_bld*omg.^2+k_bld+1i*omg.*c_bld;
        Kmat=[K11,K12;K21,K22];
        u_vect=(eye(size(Kmat))/Kmat)*[1;0];
        u_mat(1,i_o)=abs(u_vect(1));
        u_mat(2,i_o)=abs(u_vect(2));
        for i_v=1:length(vS_vect)
            K1=KF_vect(i_v);
            c1=cF_vect(i_v);
            u1_vect(i_v)=1/(-mf*omg.^2+1i*omg.*c1+K1);
            u_R(i_o,i_v)=abs(u1_vect(i_v))/abs(u_vect(2));
        end
    end

    if i_c==2
        u_matY=u_mat./u_mat(1,1);
        u_Ry=u_R;
    elseif i_c==3
        u_matZ=u_mat./u_mat(1,1);
        u_Rz=u_R;
    end
    figure
    hold on
    leg_vect = cell(1, length(f_vect1));
    for i_f = 1:length(f_vect1)
        plot(vS_vect, (u_R(i_f, :)), 'LineWidth', 1.5)
        leg_vect{i_f} = sprintf('f = %d Hz', f_vect1(i_f));
    end

    legend(leg_vect, 'Box', 'off', 'Interpreter', 'latex',...
        'FontSize', 12)
    xlabel({'Shear wave velocity (Hz)'}, 'FontSize', 16,...
        'Interpreter', 'latex')
    ylabel('$|u_{1}|/|u_{2}|$', 'FontSize', 16,...
        'Interpreter', 'latex')

    set(gca, 'XTickLabelMode', 'auto');
    set(gca, 'YTickLabelMode', 'auto');

    set(gcf, 'Units', 'inches', 'Position', [30 4 10 6],...
        'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
    filename = ['URatio_', num2str(i_c),...
        '_l', num2str(l), '_by_b', num2str(b),...
        '_ftyp_', ftyp, '_Vs_', num2str(V_s),...
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



