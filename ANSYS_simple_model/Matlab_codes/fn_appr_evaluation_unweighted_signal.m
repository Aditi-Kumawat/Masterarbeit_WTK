%Testing
%V_mat = [10,20,30,10];
%T_mat = [1,2,3,10];
%V_mat = 10;
%T_mat = 1;
%a = fn_appr_evaluation_unweighted_signal(V_mat,T_mat ,0.8)

function KB_fmax_appr = fn_appr_evaluation_unweighted_signal(V_max,freq,cf, threshold)
     % KB_fmax_appr = [computed value, lower bound (-15%), upper bound(+15%)]
     KB_fmax_appr = zeros(length(V_max),3);
     %value = zeros(length(V_max),1);
     value = (1/sqrt(2)).*(1./sqrt(1+(power((5.6 ./freq),2))).*V_max*cf);
     KB_fmax_appr(:,1) = value;
     KB_fmax_appr(:,2) = value*(1-threshold);
     KB_fmax_appr(:,3) = value*(1+threshold);
end