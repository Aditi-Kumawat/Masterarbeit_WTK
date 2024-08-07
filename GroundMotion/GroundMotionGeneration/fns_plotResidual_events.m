
function fns_plotResidual_events(LME_model)

    %% Plot in-event and between event residual
    residuals = table(LME_model.Residuals.Raw, ...
                      LME_model.Variables.M,...
                      LME_model.Variables.Dis,...
                      LME_model.Variables.Event,...
                      'VariableNames',{'res','M','Dis','Event'} );
    
    mean_r_event = grpstats(residuals, 'Event', {'mean','std'});
    overal_mean_r = mean(residuals.res);
    
    with_in = mean_r_event.mean_res - overal_mean_r;
    between = overal_mean_r  - residuals.res;
    
    
    disp(['intra = ',num2str(std(with_in))])
    disp(['inter = ',num2str(std(between))])
    figure
    subplot(211)
    scatter(residuals.M, between);
    hold on 
    scatter(mean_r_event.mean_M, with_in, 'filled')
    yline(0,'Color','[0.15,0.15,0.15]','LineStyle','--')
    xlabel("$M_{L}$", 'Interpreter', 'latex')
    ylabel('Residuals', 'Interpreter', 'latex')
    legend("Intra-event","Inter-event")
    ylim([-5,5])

    %subplot(211)
    %scatter(LME_model.Variables.Wg, between, 'filled')
    %yline(0,'Color','[0.15,0.15,0.15]','LineStyle','--')
    %xlabel("Wg")
    %ylabel("Residuals")
    %ylim([-1,1])
    
    subplot(212)
    scatter((LME_model.Variables.R),between, 'filled')
    yline(0,'Color','[0.15,0.15,0.15]','LineStyle','--')
    ylim([-2.5,2.5])
    xlabel("$R$", 'Interpreter', 'latex')
    ylabel('$\sigma$', 'Interpreter', 'latex')


    

    figure
    subplot(211)
    scatter(LME_model.Variables.Wg, between, 'filled')
    yline(0,'Color','[0.15,0.15,0.15]','LineStyle','--')
    xlabel('$\omega_{g}/2 \pi$', 'Interpreter', 'latex')
    ylabel('$\tau$', 'Interpreter', 'latex')
    ylim([-1,1])

    %subplot(211)
    %scatter(LME_model.Variables.Wg, between, 'filled')
    %yline(0,'Color','[0.15,0.15,0.15]','LineStyle','--')
    %xlabel("Wg")
    %ylabel("Residuals")
    %ylim([-1,1])
    
    subplot(212)
    scatter((LME_model.Variables.R),between, 'filled')
    yline(0,'Color','[0.15,0.15,0.15]','LineStyle','--')
    ylim([-1,1])
    xlabel("$R$", 'Interpreter', 'latex')
    ylabel('$\sigma$', 'Interpreter', 'latex')


    figure
    histogram(LME_model.residuals,'Normalization', 'pdf')
end