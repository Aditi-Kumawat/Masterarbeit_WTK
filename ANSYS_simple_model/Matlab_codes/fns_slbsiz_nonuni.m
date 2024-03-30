classdef fns_slbsiz_nonuni
    methods (Static)
        function fldr = get_fldrnm_slbsiz_nonuni(n_str,...
                n_rx, n_ry, l, b, ftyp, V_s, L_f, B_f,r_off)
            fldr = ['n_storeys_',num2str(n_str),'_n_rooms_X_',...
                num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),...
                '_l',num2str(l),'_by_b',num2str(b),'_ftyp_',ftyp,...
                '_Vs_',num2str(V_s),'_Lf_',num2str(L_f),...
                '_Bf_',num2str(B_f),'_room_off_',r_off];
        end

        %%
        function [f_vect,Uamp_mat,Ucpmlx_mat]=get_TFslbsiz_nonuni(...
                n_str, n_rx, n_ry,l_vect, b_vect, ftyp, V_s, L_f,...
                B_f,bf_nm,i_str,cmpt,n_c,r_fldr,cols,r_off)
            for i_l = 1:length(l_vect)
                l = l_vect(i_l);
                b = b_vect(i_l);
                % Get the folder name for the specific configuration
                fldr = fns_slbsiz_nonuni.get_fldrnm_slbsiz_nonuni...
                    (n_str,n_rx, n_ry,l, b, ftyp, V_s, L_f, B_f,r_off);
                fil_nm = arrayfun(@(x) sprintf(bf_nm, x{1}, i_str,...
                    l, b),cmpt, 'UniformOutput', false);

                cd ..
                cd Results_Ansys
                fil_pth = fullfile(r_fldr, fldr, fil_nm);
                U_all = cellfun(@(x) readtable(x), fil_pth,...
                    'UniformOutput', false);
                cd ..
                cd Matlab_codes

                for i_c = 1:n_c
                    U = U_all{i_c};
                    U.Properties.VariableNames = cols;
                    if i_l == 1
                        f_vect = U.Freq;
                        Uamp_mat{i_c} = U.AMPL;
                        Ur_mat{i_c} = U.REAL;
                        UIm_mat{i_c} = U.IMAG;
                        Ucpmlx_mat{i_c} = U.REAL+1i.*U.IMAG;
                    else
                        Uamp_mat{i_c}(:, i_l) = U.AMPL;
                        Ur_mat{i_c}(:, i_l) = U.REAL;
                        UIm_mat{i_c}(:, i_l) = U.IMAG;
                        Ucpmlx_mat{i_c}(:, i_l)=U.REAL+1i.*U.IMAG;
                    end
                end
            end
        end
        %%
        function plt_TFcmpr_slbsiz_nonuni(f,f_off,Uamp_1,Uamp_2,...
                i_str, n_rx, n_ry, l_vect, b_vect, ftyp,...
                V_s, L_f, B_f,i_c,cmpt,...
                room_off,l_vect_off, b_vect_off,rf_fldr)
            ha_col = @colors;
            lStyl = {'-', ':', ':', '-.'};
            lcol = {ha_col('boston university red'), ha_col('cadmium green'),...
                ha_col('black'),ha_col('denim'),...
                ha_col('dark goldenrod')};
            ustr_vect={'$u_x$~(m)','$u_y$~(m)','$u_z$~(m)'};
            figure
            for i_lb=1:length(l_vect)
                l=l_vect(i_lb);
                b=b_vect(i_lb);
                l_off=l_vect_off(i_lb);
                b_off=b_vect_off(i_lb);
                % txt_1 = ['Floor~size:~',num2str(l),'m~x',...
                % num2str(b),'m'];
                % txt_2 = ['Floor~size:~',num2str(l_off),'m~x',...
                % num2str(b_off),'m,~','Offset:~2m'];
                txt_1='Uniform~floor~size';
                txt_2='Non-uniform~floor~size';
            if i_c==1
                i_col=1;
            elseif i_c==2
                i_col=2;
            else
                i_col=4;
            end
                hold on
                plot(f,Uamp_1(:,i_lb),...
                    'linestyle',lStyl{mod(i_lb-1,numel(lStyl))+1},...
                    'DisplayName',txt_1,'LineWidth',1.5,...
                    'Color',lcol{mod(i_lb-1,numel(lcol))+i_col})
                hold on
                plot(f_off,Uamp_2(:,i_lb),...
                    'linestyle',lStyl{mod(i_lb-1,numel(lStyl))+2},...
                    'DisplayName',txt_2,'LineWidth',1.5,...
                    'Color',lcol{mod(i_lb-1,numel(lcol))+3})
            end
            if i_str==0
                ylim([0.9,1.2])
            else
                ylim([0,16])
            end
            xlim([0,40])
            %text_Str_TF=['Floor=~',num2str(i_str),...
            % ',~Transfer~function'];
            %title(text_Str_TF,'Interpreter','latex','FontSize',8)

            fns_plot.setPltProps(ustr_vect,i_c);

            filename = ['TFcmpr_slbsiz_nonuni',cmpt{i_c},...
                num2str(i_str),'_n_rooms_X_', num2str(n_rx),...
                '_n_rooms_Y_', num2str(n_ry),...
                '_ftyp_', ftyp,'_Vs_', num2str(V_s),...
                '_Lf_', num2str(L_f), '_Bf_', num2str(B_f),...
                '_room_off_',room_off '.pdf'];

            cd SAVE_FIGS
            if ~exist(rf_fldr, 'dir')
                mkdir(rf_fldr);
            end
            saveas(gcf, fullfile(rf_fldr, filename));
            cd ..
            cd ..
            cd Matlab_codes
        end
    end
end