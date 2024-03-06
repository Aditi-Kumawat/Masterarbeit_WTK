function fns_visualized_fitting(LME_model ,Var_table, Response, PredList_M, PredList_D)

    %% Plot the original data points
    M = LME_model.Variables.M;
    D = LME_model.Variables.Dis;

    % Define the range of values for the variables you want to vary
    x_values = linspace(min(M), max(M), 20); % Define the range of x values
    y_values = linspace(min(D), max(D), 20); % Define the range of y values

    figure
    scatter3(M,D,exp(Response),'filled')
    hold on 

    %% Generate the predict point
    num_data = length(M);

    for i = 1:length(PredList_M)
        Var_table.M = PredList_M(i) * ones(num_data,1);
        Var_table.D = PredList_D(i) * ones(num_data,1);
        Var_table.LnDis = log(PredList_D(i)) * ones(num_data,1);
        Pred_Response = random(LME_model,Var_table);
       
        scatter3(Var_table.M,Var_table.D,exp(Pred_Response),"*")

        Var_table.M = PredList_M(i) * ones(num_data,1);
        Var_table.D = PredList_D(i) * ones(num_data,1);
        
    end


    [X, Y] = meshgrid(x_values, y_values); % Create a grid of x and y values
    num_point = length(X(:));
    Surface_point = table(ones(num_point,1),...
                          X(:),...
                          log(Y(:)),...
                          Y(:),...
                          ones(num_point,1),...
                          ones(num_point,1),...
                          ones(num_point,1),...
                          ones(num_point,1),...
                          ones(num_point,1),...
                          ones(num_point,1),...
                          'VariableNames',{'Event','M','LnDis','Dis','D','R','lnPGA','Wg','DRg','Wc'});
    Response_Surface = exp(predict(LME_model,Surface_point));
    Response_Surface  = reshape(Response_Surface , size(X));

    
    s = surf(X,Y,Response_Surface,'FaceAlpha',0.5);
    s.EdgeColor = 'none';
  

    xlabel('M');
    ylabel('Distance');

end