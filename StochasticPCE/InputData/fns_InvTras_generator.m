function X_doe = fns_InvTras_generator(X_list,type)
    type_list = {'Gaussian', 'Uniform', 'Gamma', 'Beta'};
    
    if ismember(type, type_list)
        if strcmp(type,'Gaussian')
            X_doe = norminv(X_list,0,1);
        elseif strcmp(type,'Uniform')
            X_doe = 2*X_list - 1;
        elseif strcmp(type,'Gamma')
            error('Not avaiable yet');
        else
            error('Not avaiable yet');
        end
    else
        error('Input type is not in the list.');
    end

end