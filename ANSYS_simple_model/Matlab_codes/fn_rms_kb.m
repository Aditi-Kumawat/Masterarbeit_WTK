%Testing
%Pesuedo_matrix = 1*ones([1001,4]);
%Pesuedo_time = linspace(0, 15, 1001);

%time_interval= Pesuedo_time(3) - Pesuedo_time(2);
%exp(-(time(i)-time(j))/time_constant)

%answer = find_RMS_KBf(Pesuedo_time, Pesuedo_matrix , 0.125)

function KB_f = fn_rms_kb(time, KB , time_constant)
    KB_f = zeros(size(KB));
    if length(time) == length(KB)
        %initialized
        df_ = time(3) - time(2);       
        %compute KB_f
        for i = 1:length(KB_f)
            exp_term = transpose(exp(-(time(i)-time(1:i))/time_constant));
            KB_f(i,:) = sqrt(sum(exp_term.*((KB(1:i,:).^2)*df_),1)/time_constant); 
        end
    else
        disp("ERROR! different length")
    end
end