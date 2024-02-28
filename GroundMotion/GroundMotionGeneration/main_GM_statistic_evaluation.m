clear;
clc;
close all;

dir = 'Y';
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;

Spectral_info = data(:,2);
Time_info = data(:,3);
GM_info =   data(:,4);

i_len = Spectral_info{:,1};
j_len = Time_info{:,1};
k_len = GM_info{:,1};

cd SAVE_FIGS\FIGS_HISTO_Y_type1\

for i = 1:length(i_len)
    h = histogram(cellfun(@(x) x(i),Spectral_info),100);
    if i == 1
        title(['Histogram of ',dir,' dir input']);
        xlabel('Omega g')
        save_name = ['Histogram_',dir,'_VaryS0_Omega_g'];
    elseif i == 2
        title(['Histogram of ',dir,' dir input']);
        xlabel('Beta g')
        save_name = ['Histogram_',dir,'_VaryS0_Beta_g'];
    elseif i == 3
        title(['Histogram of ',dir,' dir input']);
        xlabel('Ratio of Omega c (Omega c/Omega g)');
        save_name = ['Histogram_',dir,'_VaryS0_Omega_c'];
    else 
        title(['Histogram of ',dir,' dir input']);
        xlabel('S0');
        save_name = ['Histogram_',dir,'_VaryS0_S0'];
    end
    savefig([save_name,'fig']);
    saveas(gcf, save_name, 'jpg');
    close;
end

for j = 1:length(j_len)
    h = histogram(cellfun(@(x) x(j),Time_info),100);
    if j == 1
        title(['Histogram of ',dir,' dir input']);
        xlabel('Time at 1% Percentile (sec)')
        save_name = ['Histogram_',dir,'_VaryS0_1p_g'];
    elseif j ==2
        title(['Histogram of ',dir,' dir input']);
        xlabel('Time at 5% Percentile (sec)')
        save_name = ['Histogram_',dir,'_VaryS0_5p_g'];
    elseif j ==3
        title(['Histogram of ',dir,' dir input']);
        xlabel('Ratio of time at 45% Percentile / Duration')
        save_name = ['Histogram_',dir,'_VaryS0_45ratio_g'];
    else
        title(['Histogram of ',dir,' dir input']);
        xlabel('Duration (sec)')
        save_name = ['Histogram_',dir,'_VaryS0_duration_g'];
    end
    savefig([save_name,'fig']);
    saveas(gcf, save_name, 'jpg');
    close;
end

for k = 1:length(k_len)-1
    h = histogram(cellfun(@(x) x(k),GM_info),100);
    if k == 1
        title(['Histogram of ',dir,' dir input']);
        xlabel('PGA')
        save_name = ['Histogram_',dir,'_VaryS0_PGA_g'];
    else
        title(['Histogram of ',dir,' dir input']);
        xlabel('Record time (sec)')
        save_name = ['Histogram_',dir,'_VaryS0_recordT_g'];
    end
    savefig([save_name,'fig']);
    saveas(gcf, save_name, 'jpg');
    close;
end

cd .. 
cd ..