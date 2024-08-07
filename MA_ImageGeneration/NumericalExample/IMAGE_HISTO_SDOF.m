clear;
clc;
close all;

%% 
% MODIFY THE UNIT FROM m/s TO mm/s
% REJECTIOIN: 20 mm/s (fns_generateTable_Scenario)
% for modification
% line 19, 203, modify the dir
% line 230, 291 modify the looping index
%%

addpath("C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\GroundMotionGeneration")
addpath("C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\GroundMotionGeneration\BaselineCorrection")


TEST_CASE = 3;
NumDOE = 1000;
NumVar = 6;
dir = 'Z';
valid_point = 750;

Learning_type = 'TR';
%Learning_type = 'VAL';

% Only use for output, irrelvent to input
output_type = "rand";
%output_type = "fix";

DOE_file_name = sprintf('TEST%d_X_SBAGM_V%d_%s_DOE_%d_DIR_%s.mat',2,NumVar,'TR',1000,'Z');
DOE_path =['C:\Users\v196m\Desktop\master_project\Masterarbeit\StochasticPCE\InputData\',DOE_file_name];
X = load(DOE_path);

%MODIFY
%Learning_type = 'VAL';

if strcmp(output_type, 'rand')

    output_file_name = sprintf('TEST%d_Y_SBAGM_V%d_%s_RAND_DOE_%d_DIR_%s.mat',TEST_CASE,NumVar,Learning_type,NumDOE,dir);
    
elseif strcmp(output_type, 'fix')
    output_file_name = sprintf('TEST%d_Y_SBAGM_V%d_%s_FIX%d_DOE_%d_DIR_%s.mat',TEST_CASE,NumVar,Learning_type,valid_point,NumDOE,dir);
end

output_path = ['.\REALIZATION\',output_file_name];


%% Import the data
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;
search_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';

% Import the data of depth
Depth_path = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\Event_list_Insheim2.txt';
% Read the text file into a table
T = readtable(Depth_path);

for j = 1:length(T.Var1)
    % Convert the string to a datetime object
    dt = datetime(T.Var1(j), 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
    % Convert the datetime object to a formatted string
    T.Var7(j) = "GM_"+ sprintf('%d_%02d_%02d_%02d_%02d_%02d', ...
        year(dt), month(dt), day(dt), hour(dt), minute(dt), second(dt));
end

Record_info = data(:,1);
Spectral_info = data(:,2);
Time_info = data(:,3);
GM_info =   data(:,4);


%% Making Info Matrix
% MAT = ( M, D, R, S0, W_g, Damp_g, W_f_ratio, Duration, T_mid_ratio)
Info_matrix = zeros(length(Record_info),10);
Info_array = [];
for i = 1:length(Record_info)
    Record_info = data(:,1);
    Record_info_i = Record_info{i};
    
    % Use fileparts to extract the folder parts
    [path_record, ~, ~] = fileparts(Record_info_i{1});
    % Extract the directory names
    [~, current_dir, ~] = fileparts(path_record);
    [~, parent_dir, ~] = fileparts(fileparts(path_record)); 
    index = find(strcmp(T.Var7, parent_dir));

    if ~isempty(index)
        D = T.Var4(index);
    else
        dips("ERROR! no depth found")
    end

    R = Record_info_i{2};
    M = Record_info_i{3};
    
    % modify the unit of input PGA from m/s^2 to mm/s^2
    PGA         = GM_info{i}(1)*1000;
    W_g         = Spectral_info{i}(1)/(2*pi);
    Damp_g      = Spectral_info{i}(2);
    W_f_ratio   = Spectral_info{i}(3);
    Duration    = Time_info{i}(4);
    T_mid_ratio = Time_info{i}(3);

    Info_matrix(i,1) = M;
    Info_matrix(i,2) = D;
    Info_matrix(i,3) = R;
    Info_matrix(i,4) = PGA;
    Info_matrix(i,5) = W_g;
    Info_matrix(i,6) = Damp_g;
    Info_matrix(i,7) = W_f_ratio;
    Info_matrix(i,8) = Duration;
    Info_matrix(i,9) = T_mid_ratio;
    Info_matrix(i,10) = index;

    Info_array = [Info_array; {parent_dir},{current_dir}, ...
         M,D,R,PGA,W_g,Damp_g,W_f_ratio,Duration,T_mid_ratio];
end


% Case 145 has too extreme value for Beta
% Case 36 has too extreme value for W_c

%% Fitting Scenario: Wc
% Fitting 
distance =  sqrt(power(Info_matrix(:,2),2)+ power(Info_matrix(:,3),2));
VarsTable= table(Info_matrix(:,10), Info_matrix(:,1),...
                 log(distance), distance, ...
                 log(Info_matrix(:,2)),Info_matrix(:,2),...
                 log(Info_matrix(:,3)),Info_matrix(:,3),...
                 log(Info_matrix(:,4)),...
                 (Info_matrix(:,5)),...
                 (Info_matrix(:,6)),...
                 (Info_matrix(:,7)),...
    'VariableNames',{'Event','M','LnDis','Dis','LnD','D','LnR','R','lnPGA','Wg','DRg','Wc'});

% Remove Outlier
VarsTable(200,:) = [];
VarsTable(153,:) = [];
VarsTable(107,:) = [];
VarsTable(105,:) = [];


LME_lnPGA = fitlme(VarsTable,'lnPGA ~  M + LnD + LnR  + Wg + (LnR|Event) +  ( Wg|Event)');
LME_lnWc  = fitlme(VarsTable, 'Wc     ~ R +  Wg + ( R|Event)+  ( Wg|Event)');
% Beta should be fixed 
LME_lnBeta = fitlme(VarsTable,'DRg  ~  R +  Wg + ( R|Event)+  ( Wg|Event)');


num_DOE = length(X.X(:,1));
if strcmp(output_type,"rand")
    % Training
    Pesuedo_M = exp(-0.378+ 0.53*X.X(:,1));
    %Pesuedo_R = 8.5+ 2.2*X.X(:,2);
    Pesuedo_R = sqrt(power(8+ 7.5*X.X(:,2),2) + power(8+ 7.5*X.X(:,3),2));
    Pesuedo_Wg = exp(2.76+ 0.37*X.X(:,4));
    X_rand = fns_generateTable_Scenario(LME_lnPGA,LME_lnWc,Pesuedo_M,Pesuedo_R,Pesuedo_Wg,0.3);
    X_rand.Wb = 12+1*X.X(:,5);
    X_rand.beta = exp(-3.2+0.1*X.X(:,6));
    Y = SDOF_simulation(X_rand,Learning_type,dir,'rand',[]);
    %save(output_path ,'Y','-mat');
    figure
    histogram(Y,'Normalization','probability','FaceColor',[0 0.4470 0.7410]);
    xlabel('$ln(v_{max})$ for $\mathbf{X}_{sb}$ (mm/s), N=500', 'Interpreter', 'latex');
    ylabel('Normalized frequency', 'Interpreter', 'latex');
    %legend('Validation sets','SPCE Prediction')
    %xlim([-7,8])
    grid on;


elseif strcmp(output_type,"fix")
    % Validation point 
    Valid_point_1 = X.X(valid_point,:).*ones(num_DOE,1);
    Pesuedo_M = exp(-0.378+ 0.53*Valid_point_1(:,1));
    %Pesuedo_R = 8.5+ 2.2*Valid_point_1(:,2);
    Pesuedo_R = sqrt(power(8+ 7.5*Valid_point_1(:,2),2) + power(8+ 7.5*Valid_point_1(:,3),2));
    Pesuedo_Wg = exp(2.76+ 0.37*Valid_point_1(:,4));
    X_valid = fns_generateTable_Scenario(LME_lnPGA,LME_lnWc,Pesuedo_M,Pesuedo_R,Pesuedo_Wg,0.3);
    X_valid.Wb = 12+1*Valid_point_1(:,5);
    X_valid.beta = exp(-3.2+0.1*Valid_point_1(:,6));

    Y = SDOF_simulation(X_valid,Learning_type,dir,'fix',valid_point);
    %save(output_path ,'Y','-mat');
end



function Y = SDOF_simulation(X, learning_type,dir_ANSYS, output_type,fixpoint)

       %% Import data from ANSYS
       % Define the number of storeys, rooms in x- y-direction
       n_str = 3;
       n_rx = 3;
       n_ry = 4;
       % Define the type of foundation as either 'PLATE' or 'FOOTING'
       ftyp = 'FOOTING';
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
       %DR_index = 1;
       dir = dir_ANSYS;
       floor_num = 2;
    
       %% Z dir
       % log(PGA) ~ Normal(-8.25,1.4)
       % Wg ~ LogNormal(2.5,0.3)
       % Beta = 0.3
       % W_c_ratio ~ LogNormal(-1.2,0.4)
       % Wn ~ Normal(12,1)

       PGA = exp(X.lnPGA);
       W_g = 2*pi*X.Wg;
       beta = X.DRg;
       W_c_ratio = X.Wc;
       
       Wn = 2*pi*X.Wb;
       %DR = 0.05;
       DR = X.beta;
       %Y = zeros(length(X.lnPGA),1);
       Y = zeros(500,1);
       if strcmp(output_type,"rand") 
           %figure
  %MODIFY
           % for training, start from 2 to 501
           %for i = 2:501
           % for validation, start from 501 to 1000
           for i = 2:501
               disp(['Case ',num2str(i)])
               disp(['     Parameter: W_g = ',num2str(W_g(i)), ',W_c = ',num2str(W_c_ratio(i))])
               disp(['     Parameter: PGA = ',num2str(PGA(i)), ',W_n = ',num2str(Wn(i))])
               disp(['     Parameter: DR = ',num2str(DR(i))])
               
               GM_params = [W_g(i), beta(i) ,W_c_ratio(i)];
               [Time, ~, Vel, ~] = fns_generateGM_Params([], [], 100 ,"Hu_S0", GM_params, [0.5, 1 ,4, 10], PGA(i));
               
               % FFT half-positive
               L = length(Vel);
               Fs = round(1/(Time(3)-Time(2)),200);
               data_FFT = fft(Vel);
               P2 = data_FFT;
               P1 = P2(1:floor(L/2+1));
               f = Fs*(0:(L/2))/L;
           
               % ANSYS FRF, excited by Displacement
               %Omega = 2*pi*f;
               DR_index = i;
               folder_name = ['C:/Users/v196m/Desktop/master_project/Masterarbeit/ANSYS_simple_model/Results_Ansys/DataFromServer/n_storeys_',num2str(n_str),'_n_rooms_X_',num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),'_ftyp_',ftyp,'_Vs_',num2str(V_s),'_Lf_',num2str(L_f),'_Bf_',num2str(B_f),'_DR_',num2str(DR_index)];
               file_name = ['Disp_Center_',dir,'_',num2str(floor_num),'.csv'];
               path = fullfile(folder_name,file_name );
                
               FRF = readtable(path);
               FRF.Freq(1) = 0;
               FRF_R = interp1(FRF.Freq,FRF.REAL,f);
               FRF_I = interp1(FRF.Freq,FRF.IMAG,f);
               FRF_complex = FRF_R + 1i*FRF_I;
    
               % Differential, Disp FRF -> Vel FRF
               FRF_vel_complex = 2*pi*1i*FRF_complex.*f;

               % Compute the Response
               Resp_f = P1.*transpose(FRF_vel_complex);
    
               % IFFT from half-positive
               Resp_f_pad = [Resp_f; conj(flipud(Resp_f(2:end-1,:)))];
               Resp_ifft = ifft(Resp_f_pad*Fs, L, 1, 'symmetric');
               Resp_ifft = Resp_ifft(1:L,:)/Fs;
                
 %MODIFY        
               if strcmp(learning_type,'TR')
                    Y(i-1) = log(max(abs(Resp_ifft))); 
               else
                    Y(i-500) = log(max(abs(Resp_ifft))); 
               end
           end

       elseif strcmp(output_type,"fix")

           % ANSYS FRF, excited by Displacement
           DR_index = fixpoint;
           folder_name = ['C:/Users/v196m/Desktop/master_project/Masterarbeit/ANSYS_simple_model/Results_Ansys/DataFromServer/n_storeys_',num2str(n_str),'_n_rooms_X_',num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),'_ftyp_',ftyp,'_Vs_',num2str(V_s),'_Lf_',num2str(L_f),'_Bf_',num2str(B_f),'_DR_',num2str(DR_index)];
           file_name = ['Disp_Center_',dir,'_',num2str(floor_num),'.csv'];
           path = fullfile(folder_name,file_name );
           FRF = readtable(path);
           
           % for training, start from 2 to 501
           %for i = 2:501
           % for validation, start from 501 to 1000
           for i = 501:1000
               disp(['Case ',num2str(i)])
               disp(['     Parameter: W_g = ',num2str(W_g(i)), ',W_c = ',num2str(W_c_ratio(i))])
               disp(['     Parameter: PGA = ',num2str(PGA(i)), ',W_n = ',num2str(Wn(i))])
               disp(['     Parameter: DR = ',num2str(DR(i))])
               
               GM_params = [W_g(i), beta(i) ,W_c_ratio(i)];
               [Time, ~, Vel, ~] = fns_generateGM_Params([], [], 100 ,"Hu_S0", GM_params, [0.5, 1 ,4, 10], PGA(i));
               
               % FFT half-positive
               L = length(Vel);
               Fs = round(1/(Time(3)-Time(2)),200);
               data_FFT = fft(Vel);
               P2 = data_FFT;
               P1 = P2(1:floor(L/2+1));
               f = Fs*(0:(L/2))/L;

               FRF.Freq(1) = 0;
               FRF_R = interp1(FRF.Freq,FRF.REAL,f);
               FRF_I = interp1(FRF.Freq,FRF.IMAG,f);
               FRF_complex = FRF_R + 1i*FRF_I;
               % Differential, Disp FRF -> Vel FRF
               FRF_vel_complex = 2*pi*1i*FRF_complex.*f;
               
               % Compute the Response
               Resp_f = P1.*transpose(FRF_vel_complex);
    
               % IFFT from half-positive
               Resp_f_pad = [Resp_f; conj(flipud(Resp_f(2:end-1,:)))];
               Resp_ifft = ifft(Resp_f_pad*Fs, L, 1, 'symmetric');
               Resp_ifft = Resp_ifft(1:L,:)/Fs;
               
               if strcmp(learning_type,'TR')
                    Y(i-1) = log(max(abs(Resp_ifft))); 
               else
                    Y(i-500) = log(max(abs(Resp_ifft))); 
               end
           end
       end

end