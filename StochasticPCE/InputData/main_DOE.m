clear;
clc;

%% Set the number of samples
TEST_CASE = 1;
dir = 'Z';
numSamples = 1000;
numVars = 6;

Learning_type = 'TR';
%Learning_type = 'VAL';

%DR_ANSYS = 5;

type_list = ["Gaussian","Uniform","Uniform","Gaussian","Gaussian","Gaussian"];

%% DOE
lhsSamples = lhsdesign(numSamples, numVars);
X = zeros(numSamples, numVars);

for i = 1:numVars
    X_i = fns_InvTras_generator(lhsSamples(:, i),type_list(i));
    X(:,i) = X_i;

    
end
%X_DR_ANSYS= exp(-3.2+0.1*X(:,DR_ANSYS));

Pesuedo_R = sqrt(power(8+ 7.5*X(:,2),2) + power(8+ 7.5*X(:,3),2));
histogram(Pesuedo_R,10,'Normalization','probability','FaceColor',[0 0.4470 0.7410]);
xlabel('$R = \sqrt{R_{x}^{2}+R_{y}^{2}}$ (km)', 'Interpreter', 'latex')
ylabel('Normalized frequency', 'Interpreter', 'latex')
grid on

%fid = fopen('DAMPR.txt', 'w');
%fprintf(fid, ' %14.7e \n', X_DR_ANSYS');
%fclose(fid);
%type 'DAMPR.txt'

%output_file_name = sprintf('TEST%d_X_SBAGM_V%d_%s_DOE_%d_DIR_%s.mat',TEST_CASE,numVars,Learning_type,numSamples,dir);

%save(output_file_name,'X','-mat');


