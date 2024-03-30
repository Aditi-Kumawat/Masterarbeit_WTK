%% Initialization
clear;clc;close all;


% Define the number of storeys, rooms in x- y-direction
n_str = 1;
n_rx = 1;
n_ry = 1;

% Define the length, width, and height of the building
l_vect=[2 3 4 5 6 7 8];
b_vect=[2 3 4 5 6 7 8];
% l_vect=[3 5 7];
% b_vect=[3 5 7];
h = 3;

% Define the type of foundation as either 'PLATE' or 'FOOTING'
ftyp = 'FOOTING';

% Define the velocity of the excitation
V_s = 100;

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
name_evnt='Poing';
%%
if strcmp(name_evnt, 'Poing')
    evnt='Po2016';
    %stn_vect={'POI01', 'POI02', 'POI03'};
    stn_vect={'POI01'};
    date='2016_12_20';
    time='03_30_51';
elseif strcmp(name_evnt, 'Unterhaching')
    evnt='Part1';
    stn_vect={'UH1', 'UH2', 'UH3'};
    date='2013_04_16';
    time='21_51_42';
end

n_stns=length(stn_vect);



%% Importing Data
bf_nm_u = 'fftd_%d_%s_%s_%s';
bf_nm_v = 'fftv_%d_%s_%s_%s'
nzero=1;   % what is nzero for? nzero = 6
cols = {'Freq', 'Re','Im','Amp'};
s_dir = [1 2 3];
n_snr = numel(s_dir);
%% Importing Transfer Function
%rf_fldr = 'MultiUnitBld_GeomVary';
rf_fldr = 'UnitBld_GeomVary';
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';
cols1 = {'Freq', 'AMPL','PHASE','REAL','IMAG'};
cmpt = {'X', 'Y', 'Z'};

%%
bf_nm_vt = 'v_%d_%s_%s_%s';
cols_t = {'tim', 'val'};

%% KB(f) Filter
% referecne frequency = 5.6
% DIN 4150-2
get_KB_freq = @(V_freq,freq) V_freq./(sqrt(1+power((5.6 ./freq),2)));


for i_stn=1:n_stns
    stn=stn_vect{i_stn};
    display(stn)
    if strcmp(name_evnt, 'Poing')
        ff_fldr = fullfile('GM','GM_POI2016',stn);
    elseif strcmp(name_evnt, 'Unterhaching')
        fldr_nm = [stn, '_', evnt];
        ff_fldr = fullfile('GM', 'GM_UH',fldr_nm);
    end
    %% Importing sensor data
    %Poing data 
    %   start from Ln 6 and end at Ln 2054, length = 2049 
    %f_inpt = frequnecy, length = 2049
    %ff_Uamp_mat = amplitude
    %ff_Ur_mat = real
    %ff_UIm_mat= imag
    %ff_Ucmplx_mat = complex number, combined real and imag
    [f_inpt,ff_Uamp_mat,ff_Ur_mat,ff_UIm_mat,ff_Ucmplx_mat]=...
        fns_imprtdata.get_ff_inpt(bf_nm_u,s_dir,...
        stn,date, time,n_snr,ff_fldr,cols);
    [f_inpt_V,ff_Vamp_mat,ff_Vr_mat,ff_VIm_mat,ff_Vcmplx_mat]=...
        fns_imprtdata.get_ff_inpt(bf_nm_v,s_dir,...
        stn,date, time,n_snr,ff_fldr,cols);
    % fns_plot.plt_ff(f_inpt_V, ff_Vamp_mat,bf_nm_v,123,...
    %     stn,date, time,'Velocity~(m/s)','initial')

    [t_in,ff_Vt]=...
        fns_imprtdata.get_ff_tim(bf_nm_vt,s_dir,...
        stn,date, time,n_snr,ff_fldr,cols_t);
    t_in=t_in{1};

    n_c = length(cmpt);
    Vabs_zCell = cell(1, n_str+1);
    fref_cmplx_mat_1=cell(1, n_c);
    DISP_cmplx_mat=cell(1, n_c);
    VEL_abs_mat=cell(1, n_c);
    fref_vel_amp_mat_1=cell(1, n_c);
    v_ref=5e-8;

    %% Importing Transfer function computed from ANSYS
    %data extract from simulation: [501,28], 
    %   501 rows along frequency
    %   28 columns based on Vs
    %f_vect = frequency
    %TF_amp_mat = amplitude
    %TF_cpmlx_mat = real,imag (complex number)
    for i_str = 0:n_str
        [f_vect,TF_amp_mat,TF_cpmlx_mat,lb_combs]=...
            fns_scatter.get_TF_scatter(n_str, n_rx, n_ry,...
            l_vect, b_vect, ftyp, V_s, L_f, B_f,...
            bf_nm,i_str,cmpt,n_c,rf_fldr,cols1);
        for i_c = 1:n_c
            %% Calculating Velocity
            % using interpolation maping data from len = 501 to len = 2049
            TFcpmlx_intrp{i_c}=interp1(f_vect,TF_cpmlx_mat{i_c},...
                f_inpt{i_c},'linear','extrap');
            
            % U(f) = 2*Vsensor(f)*TF(f)
            Ucmplx_mat{i_c}=2*ff_Ucmplx_mat{i_c}.*TFcpmlx_intrp{i_c};
            
            % V(f) = 2*pi*f*i*U(f) = i*omega*U(f)
            Vabs_mat{i_c}=abs(Ucmplx_mat{i_c}.*1i.*f_inpt{i_c}*2*pi);
            Vcmplx_mat{i_c}=(Ucmplx_mat{i_c}.*1i.*f_inpt{i_c}*2*pi);

            %Calculate DB
            Vss_mat{i_c}=20*log10(Vabs_mat{i_c}(nzero:end,:)./v_ref);

            %Apply filter transfer to KB(f)
            %V_KB_freq{i_c} = get_KB_freq(Vcmplx_mat{i_c},f_inpt{i_c});
            V_KB_freq{i_c} = fn_kb_highpass(f_inpt{i_c},Vcmplx_mat{i_c},5.6);
        end
        %Original data
        Vss_zCell{i_str+1} = Vcmplx_mat{3};
        Vss_xCell{i_str+1} = Vcmplx_mat{1};
        Vss_yCell{i_str+1} = Vcmplx_mat{2};

        %Data with filter: KB(f)
        V_KB_freq_zCell{i_str+1} = V_KB_freq{3};
        V_KB_freq_xCell{i_str+1} = V_KB_freq{1};
        V_KB_freq_yCell{i_str+1} = V_KB_freq{2};
        
        %Corresponding Frequency
        %freq=f_inpt_V{3}(nzero:end);
        freq_=f_inpt_V{3}(nzero:end);
    end
    
    %% Check whether the filter works or not
    %figure
    %plot(imag(Vss_zCell{1}(:,1)))
    %hold on
    %plot(imag(V_KB_freq_zCell{1}(:,1)))

    




    %% IFFT compute v(t)
    %sampling rate = 200
    Fs = 200;
    %Exponent of next higher power of 2
    %t_in = 15sec (3001 elements)
    
    %MODIFICATION 01102023
        %nfft = 2^nextpow2(length(t_in));   % nfft = 4096 
        %Fs = nfft/(t_in(end)-t_in(1));
        %t_in = transpose(linspace(t_in(1),t_in(end),nfft));
        
        nfft = length(t_in);
        
    %frequnecy array 
    freq = Fs / 2 * linspace(0, 1, nfft/2+1); %length = nfft/2+1
    dfz=freq(3)-freq(2);

        %% Verification ----------------------------------------------------------------
        % Example velocity function (you should replace this with your actual velocity data)
        velocity = sin(2 * pi * 4 * t_in) + 0.5 * sin(2 * pi * 10 * t_in);
        % Calculate the Fourier Transform of velocity
        velocity_fft = fft(velocity)/Fs;
        velocity_fft_trun = velocity_fft(1:nfft/2+1);
        %figure
        %plot(freq,abs(velocity_fft_trun));
       
        %% ----------------------------------------------------------------

    num_lb=length(lb_combs);
    for i_flur = 1:n_str+1
        %% IFFT original data

            %% Verification ---------------------------------------------------
                velocity_fft_pad = [velocity_fft_trun ; conj(flipud(velocity_fft_trun(2:end-1,:)))];
                %velocity_ifft  = 0.5*ifft(velocity_fft_pad*Fs, nfft, 1, 'symmetric');
                %velocity_ifft = velocity_ifft(1:length(t_in),:);
                velocity_ifft  = ifft(velocity_fft_pad*Fs, nfft, 1, 'symmetric');
                velocity_ifft = velocity_ifft(1:length(t_in),:);
            %% ----------------------------------------------------------------

        Vss_Zmat = Vss_zCell{i_flur};  %length = 2049
        Vss_Xmat = Vss_xCell{i_flur};
        Vss_Ymat = Vss_yCell{i_flur};
        
        %The frequency Spectra only positive-half of frequency range?
        %compute the velocity v(t) by ifft
        Vzss_fft_pad = [Vss_Zmat; conj(flipud(Vss_Zmat(2:end-1,:)))];
        Vxss_fft_pad = [Vss_Xmat; conj(flipud(Vss_Xmat(2:end-1,:)))];
        Vyss_fft_pad = [Vss_Ymat; conj(flipud(Vss_Ymat(2:end-1,:)))];

        %MODIFICATION 01102023
            %Vz_ifft = 0.5*ifft(Vzss_fft_pad*Fs, nfft, 1, 'symmetric'); %length = 4096
            %Vx_ifft = 0.5*ifft(Vxss_fft_pad*Fs, nfft, 1, 'symmetric');
            %Vy_ifft = 0.5*ifft(Vyss_fft_pad*Fs, nfft, 1, 'symmetric');
            %Vz_ifft = Vz_ifft(1:length(t_in),:);  %length = length(t_in)
            %Vx_ifft = Vx_ifft(1:length(t_in),:);
            %Vy_ifft = Vy_ifft(1:length(t_in),:);

            Vz_ifft = ifft(Vzss_fft_pad*Fs, nfft, 1, 'symmetric');
            Vx_ifft = ifft(Vxss_fft_pad*Fs, nfft, 1, 'symmetric');
            Vy_ifft = ifft(Vyss_fft_pad*Fs, nfft, 1, 'symmetric');
            Vz_ifft = Vz_ifft(1:length(t_in),:);
            Vx_ifft = Vx_ifft(1:length(t_in),:);
            Vy_ifft = Vy_ifft(1:length(t_in),:);

        % Store the result in Cell : {floor num}
        Vz_ifft_cell{i_flur} = Vz_ifft;
        Vx_ifft_cell{i_flur} = Vx_ifft;
        Vy_ifft_cell{i_flur} = Vy_ifft;


        %% IFFT data with filter
        V_KB_Zmat = V_KB_freq_zCell{i_flur};
        V_KB_Xmat = V_KB_freq_xCell{i_flur};
        V_KB_Ymat = V_KB_freq_yCell{i_flur};
        Vz_KB_fft_pad = [V_KB_Zmat; conj(flipud(V_KB_Zmat(2:end-1,:)))];
        Vx_KB_fft_pad = [V_KB_Xmat; conj(flipud(V_KB_Xmat(2:end-1,:)))];
        Vy_KB_fft_pad = [V_KB_Ymat; conj(flipud(V_KB_Ymat(2:end-1,:)))];

        %MODIFICATION 01102023
            %Vz_KB_ifft = 0.5*ifft(Vz_KB_fft_pad*Fs, nfft, 1, 'symmetric');
            %Vx_KB_ifft = 0.5*ifft(Vx_KB_fft_pad*Fs, nfft, 1, 'symmetric');
            %Vy_KB_ifft = 0.5*ifft(Vy_KB_fft_pad*Fs, nfft, 1, 'symmetric');
            %Vz_KB_ifft = Vz_KB_ifft(1:length(t_in),:);
            %Vx_KB_ifft = Vx_KB_ifft(1:length(t_in),:);
            %Vy_KB_ifft = Vy_KB_ifft(1:length(t_in),:);
            Vz_KB_ifft = ifft(Vz_KB_fft_pad*Fs, nfft, 1, 'symmetric');
            Vx_KB_ifft = ifft(Vx_KB_fft_pad*Fs, nfft, 1, 'symmetric');
            Vy_KB_ifft = ifft(Vy_KB_fft_pad*Fs, nfft, 1, 'symmetric');
            Vz_KB_ifft = Vz_KB_ifft(1:length(t_in),:);
            Vx_KB_ifft = Vx_KB_ifft(1:length(t_in),:);
            Vy_KB_ifft = Vy_KB_ifft(1:length(t_in),:);

        % Store the result in Cell : {floor num}
        Vz_KB_ifft_cell{i_flur} = Vz_KB_ifft;
        Vx_KB_ifft_cell{i_flur} = Vx_KB_ifft;
        Vy_KB_ifft_cell{i_flur} = Vy_KB_ifft;

        t = (0:length(Vz_ifft)-1) / Fs;

        max_Vxmat(i_stn,:,i_flur)=max(Vx_ifft);
        max_Vymat(i_stn,:,i_flur)=max(Vy_ifft);
        max_Vzmat(i_stn,:,i_flur)=max(Vz_ifft);
        %max_Vx_KB_mat(i_stn,:,i_flur)=max(Vx_KB_ifft);
        %max_Vy_KB_mat(i_stn,:,i_flur)=max(Vy_KB_ifft);
        %max_Vz_KB_mat(i_stn,:,i_flur)=max(Vz_KB_ifft);
        
        KB_f_x = fn_rms_kb(t, Vx_KB_ifft*1000 , 0.125);
        KB_f_y = fn_rms_kb(t, Vy_KB_ifft*1000 , 0.125);
        KB_f_z = fn_rms_kb(t, Vz_KB_ifft*1000 , 0.125);

        max_Vx_KB_f_mat(i_stn,:,i_flur)=max(KB_f_x);
        max_Vy_KB_f_mat(i_stn,:,i_flur)=max(KB_f_y);
        max_Vz_KB_f_mat(i_stn,:,i_flur)=max(KB_f_z);

        for ilb=1:num_lb

            % find the freq of max Vss
            Vss_Xvect=abs(Vss_Xmat(:,ilb));
            [value, index] = max(Vss_Xvect);
            f_x_max(i_stn,ilb,i_flur)=freq(index);
            Vss_Yvect=abs(Vss_Ymat(:,ilb));
            [value, index] = max(Vss_Yvect);
            f_y_max(i_stn,ilb,i_flur)=freq(index);
            Vss_Zvect=abs(Vss_Zmat(:,ilb));
            [value, index] = max(Vss_Zvect);
            f_z_max(i_stn,ilb,i_flur)=freq(index);


            %find the time of max Kb_f
            KB_f_x_vect=KB_f_x(:,ilb);
            [value, index] = max(KB_f_x_vect);
            t_x_max(i_stn,ilb,i_flur)=t(index);

            KB_f_y_vect=KB_f_y(:,ilb);
            [value, index] = max(KB_f_y_vect);
            t_y_max(i_stn,ilb,i_flur)=t(index);

            KB_f_z_vect=KB_f_z(:,ilb);
            [value, index] = max(KB_f_z_vect);
            t_z_max(i_stn,ilb,i_flur)=t(index);
        end
    end
end


ylbl_vect={'$v_{x,max}$,~m/s', '$v_{y,max}$,~m/s', '$v_{z,max}$,~m/s'};
ylbl=ylbl_vect{1}
cmp=cmpt{1};
fns_unitgeomdb.plot_DIN4150_3_XYZ(v_ref,n_str+1,f_x_max,max_Vxmat,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)

ylbl=ylbl_vect{2}
cmp=cmpt{2};
fns_unitgeomdb.plot_DIN4150_3_XYZ(v_ref,n_str+1,f_y_max,max_Vymat,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)

ylbl=ylbl_vect{3}
cmp=cmpt{3};
fns_unitgeomdb.plot_DIN4150_3_XYZ(v_ref,n_str+1,f_z_max,max_Vzmat,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)



criteria = fn_A_value_evaluating_Kbf("ReinesWohngebiet","day");
ylbl_vect={'$KB{f,x}$', '$KB{f,y}$', '$KB{f,z}$'};
ylbl=ylbl_vect{1}
cmp=cmpt{1};
fns_unitgeomdb.plot_DIN4150_2_XYZ(criteria,n_str+1,t_x_max,max_Vx_KB_f_mat,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)

ylbl=ylbl_vect{2}
cmp=cmpt{2};
fns_unitgeomdb.plot_DIN4150_2_XYZ(criteria,n_str+1,t_y_max,max_Vy_KB_f_mat,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)

ylbl=ylbl_vect{3}
cmp=cmpt{3};
fns_unitgeomdb.plot_DIN4150_2_XYZ(criteria, n_str+1,t_z_max,max_Vz_KB_f_mat,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)





%%% Evalutation for Z-dir for top floor
%% Comparing the KB_f_max and KB_f_max_appr from different manners
%% Vz_KB_ifft = [m/s]
%KB_f = fn_rms_kb(t, Vz_KB_ifft*1000 , 0.125);
%
%%finding the freq where maximum exist
%% Vss_zCell{2} for only top floor
%[V_max_freq,index] = max(abs(Vss_zCell{2}));
%freq_max = freq(index);
%%freq_max = transpose(freq_(index-nzero));
%
%%finding maximum unweighted velocity V_max in time-domain 
%V_max_time = max(Vz_ifft)*1000;
%%V_max_freq2time = V_max_freq*(2*pi)*1000;
%
%%Computing the appr.KB_f_max
%KB_f_max_appr = fn_appr_evaluation_unweighted_signal(V_max_time,freq_max,0.7,0.15);
%
%%Checking KB_f_max and KB_f_max_appr are matched or not
%KB_f_max = max(KB_f);
%
%disp("------------------------------------------------------")
%disp("Comparing data if the KB_f_max in range of KB_f_max_appr (+/-15%)")
%count = 0;
%for case_ = 1: length(KB_f_max)
%    if KB_f_max(case_) >= KB_f_max_appr(case_,2) && KB_f_max(case_) <= KB_f_max_appr(case_,3)
%        disp(['Case ',num2str(case_ ),': passed']);
%        count = count +1 ;
%    else
%        disp(['Case ',num2str(case_ ),': failed']);
%        disp(['  KB_f_max      = ',num2str(KB_f_max(case_))]);
%        disp(['  KB_f_max_appr = [',num2str(KB_f_max_appr(case_,2)),' ,',num2str(KB_f_max_appr(case_,3)),']']);
%    end
%end
%disp("------------------------------------------------------")
%disp(['passed ratio: ',num2str(count),'/',num2str(length(KB_f_max)),' percentage: ',num2str(count/length(KB_f_max))])



%%% Comparing with A value
%% setting criteria
%disp("------------------------------------------------------")
%disp("A value criteria test")
%criteria = fn_A_value_evaluating_Kbf("ReinesWohngebiet","day");
%for case_A = 1:length(KB_f_max)
%    A_test = fn_evaluate_A_criteria(KB_f_max(case_A),0,criteria,false);
%    disp(strcat(['Case ',num2str(case_A ),':  '], A_test));
%end


%for plot_num = 1:1
%    figure
%    subplot(1,2,1)
%    plot(t,Vz_ifft(:,plot_num))
%    hold on
%    plot(t,Vz_KB_ifft(:,plot_num))
%    title('Result under time-domain')
%    xlabel('time(s)')
%    ylabel('v(m/s)')
%    legend ('unweighted','weighted')
%    
%    subplot(1,2,2)
%    plot(freq,abs(Vss_zCell{2}(:,plot_num)))
%    hold on
%    plot(freq,abs(V_KB_freq_zCell{2}(:,plot_num)))
%    title('Result under frequency-domain')
%    xlabel('frequency(Hz)')
%    ylabel('Amplitude(m/f)')
%    legend ('unweighted','weighted')
%end


%figure
%plot(freq_,abs(Vss_zCell{2}(nzero:end,23)))
%hold on
%plot(freq_,abs(V_KB_freq_zCell{2}(nzero:end,23)))
%title('Result under frequency-domain')
%xlabel('frequency(Hz)')
%ylabel('Amplitude(m/f)')
%legend ('unweighted','weighted')



%% verification, QUESTION: using 2^nextpow2 
figure
plot(t_in,velocity);
hold on
plot(t_in,velocity_ifft,'--');

%% verification, QUESTION: using 2^nextpow2
figure
plot(freq,abs(velocity_fft_trun));

%% QUESTION: the ratio of IFFT 
%figure
%plot(t_in,Vz_ifft(:,1))



%% QUESTION
% 1. the length of t_in and the length of nfft are different
% 2. the ratio between ampl in freq-domain and its value in time-domain
%


