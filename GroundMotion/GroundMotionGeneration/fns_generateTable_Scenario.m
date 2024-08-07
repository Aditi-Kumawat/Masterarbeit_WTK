function  VarTable = fns_generateTable_Scenario(LME_PGA,LME_Wc,M_list,R_list,Wg_list,Beta_const)

    num_Y = length(M_list);
    
    % Not meaningful, just assume there is 60 events
    Event_list = randi(60,num_Y,1);
    
    %Initialized_vars_table
    Vars_init = ones(num_Y,1);

    VarTable= table(Event_list,Vars_init, ...
                    Vars_init,Vars_init,...
                    Vars_init,Vars_init,...
                    Vars_init,Vars_init,...
                    Vars_init,Vars_init,...
                    Vars_init,Vars_init,...
                    'VariableNames',{'Event','M','LnDis','Dis','LnD','D','LnR','R','lnPGA','Wg','DRg','Wc'});
    VarTable.M   = M_list;
    VarTable.R = R_list;
    VarTable.LnR = log(R_list);
    VarTable.Wg  = Wg_list;
    VarTable.DRg = Beta_const*Vars_init;

    % Prediction by regreesion
    PGA_reg = predict(LME_PGA,VarTable);
    Wc_reg = predict(LME_Wc,VarTable);
    

    % Prediction with uncertainty
    Residuals = [LME_PGA.residuals, LME_Wc.residuals];
    Cov_matrix = cov(Residuals);
    mu = [0,0];

    PGA = zeros(num_Y,1);
    Wc= zeros(num_Y,1);
    total = 0;
    for i = 1:num_Y
        R = mvnrnd(mu,Cov_matrix ,1);
        PGA(i) = PGA_reg(i)+R(1);
        Wc(i) = Wc_reg(i)+R(2);
        
        % maximum = 20 mm/s 
        while Wc(i) < 0.01 || Wc(i) >= 1 || exp(PGA(i))> 0.02*1000
            R = mvnrnd(mu,Cov_matrix ,1);
            PGA(i) = PGA_reg(i)+R(1);
            Wc(i) = Wc_reg(i)+R(2);
            total = total +1;
        end
    end

    VarTable.lnPGA = PGA;
    VarTable.Wc = Wc;
    total
end
    
