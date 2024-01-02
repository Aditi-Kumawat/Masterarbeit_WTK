function [T,Spa,Spv,Sd] = fns_response_spectra(dt,Ag,zet,g,endp)
    %%% Elastic Response Spectra, Version 2 
    % This function generates elastic response specra including Displacement
    % Spectrum, Pseudo Acceleration Spectrum, and Pseudo Velocity Spectrum which
    % are needed in a "Response Spectrum Analysis" of structures. To solve 
    % the "equation of motions" for different periods, the Newmark Linear Method 
    % was used. 
    %% Update Note:  It was clarified that Ag has the acceleration unit.
    %% (c) Mostafa Tazarv, South Dakota State University, May 2019
    %  https://sites.google.com/people.unr.edu/mostafa-tazarv/home
    %% SPEC Function Help:
    % INPUT:
    % dt:     Time Interval (Sampling rate) of the Ground Motion
    % Ag:     Ground Motion Acceleration in the unit of the "acceleration", e.g. m/s^2  
    % zet:    Damping Ratio in percent (%); e.g. 5
    % g:      Gravitational Constant; e.g. 9.81 m/s^2; g determines the output unit
    % endp:   End Period of the Spectra; e.g. 4 sec.
    % OUTPUT:
    % T:      Period of the Structure (sec.)
    % Spa:    Elastic Pseudo Acceleration Spectrum in g
    % Spv:    Elastic Pseudo Velocity Spectrum in the unit of velocity (e.g. m/s)
    % Sd:     Elastic Displacement Spectrum, in the unit of displacement (e.g. m)
    u=zeros(length(Ag),1);
    v=zeros(length(Ag),1);
    ac=zeros(length(Ag),1);
    Ag(end+1)=0;
    T(1,1)=0.00;
    for j=1:round(endp/dt)                          % equation of motion(Newmark linear method)
        omega(j,1)=2*pi/T(j);      % Natural Frequency
        m=1;       
        k=(omega(j))^2*m;
        c=2*m*omega(j)*zet/100;
        K=k+3*c/dt+6*m/(dt)^2;
        a=6*m/dt+3*c;
        b=3*m+dt*c/2;    
      for i=1:length(u)-1
         u(1,1)=0;                      %initial conditions
         v(1,1)=0;
         ac(1,1)=0;    
         df=-(Ag(i+1)-Ag(i))+a*v(i,1)+b*ac(i,1);  % delta Force
         du=df/K;
         dv=3*du/dt-3*v(i,1)-dt*ac(i,1)/2;
         dac=6*(du-dt*v(i,1))/(dt)^2-3*ac(i,1);
         u(i+1,1)=u(i,1)+du;
         v(i+1,1)=v(i,1)+dv;
         ac(i+1,1)=ac(i,1)+dac;     
      end
        Sd(j,1)=max(abs((u(:,1))));
        %Sv(j,1)=max(abs(v));
        %Sa(j,1)=max(abs(ac))/g;
        Spv(j,1)=Sd(j)*omega(j);
        Spa(j,1)=Sd(j)*(omega(j))^2/g;
        T(j+1,1)=T(j)+dt;
    end
    Ag(end)=[];
    T(end)=[];
    Sd(2,1)=0; Spv(1:2,1)=0; Spa(1:2,1)=max(abs(Ag))/g;
    %%% Plot Spectra
    %subplot(2,1,1)
    % %figure('Name','Spectral Displacement','NumberTitle','off')
    % semilogx(T,Sd,'LineWidth',2.)
    % grid on
    %xlabel('Period (sec)','FontSize',13);
    %ylabel('Sd (mm)','FontSize',13);
    %title('Displacement Spectrum','FontSize',13)
    %subplot(2,1,2)
    % %figure('Name','Pseudo Acceleration Spectrum','NumberTitle','off')
    % semilogx(T,Spa,'LineWidth',2.)
    % grid on
    %xlabel('Period (sec)','FontSize',13);
    %ylabel('Spa (g)','FontSize',13);
    %title('Pseudo Acceleration Spectrum','FontSize',13)
end