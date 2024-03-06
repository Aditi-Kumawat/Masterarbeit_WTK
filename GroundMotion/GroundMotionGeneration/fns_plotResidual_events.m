
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
    
    figure
    subplot(211)
    scatter(residuals.M, between);
    hold on 
    scatter(mean_r_event.mean_M, with_in, 'filled')
    yline(0,'Color','[0.15,0.15,0.15]','LineStyle','--')
    xlabel("M")
    ylabel("Residuals")
    legend("with-in events","between events")
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
    ylim([-5,5])
    xlabel("R")
    ylabel("Residuals")

    figure
    histogram(LME_model.residuals,'Normalization', 'pdf')
end