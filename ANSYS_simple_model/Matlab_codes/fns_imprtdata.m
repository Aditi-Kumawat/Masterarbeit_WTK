%%
classdef fns_imprtdata
    methods (Static)
        %%
        function [f,sigamp_mat,sigR_mat,sigIm_mat,sigcmplx_mat]=...
                get_ff_inpt(bf_nm_1,s_dir,...
                stn,date, time,n_s,ff_fldr,cols)

            %             sigamp_mat = cell(n_s, 1);
            f = cell(n_s, 1);
            fnm_ff = arrayfun(@(x) sprintf(bf_nm_1, x, stn, date, time),...
                s_dir, 'UniformOutput', false);
            cd ..

            fil_path = fullfile(ff_fldr, fnm_ff);

            ff_all = cellfun(@(x) readtable(x, 'ReadVariableNames',...
                false, 'HeaderLines', 5),...
                fil_path, 'UniformOutput', false);
            cd Matlab_codes
            for i_s = 1:n_s
                ff_snsr = ff_all{i_s};
                ff_snsr.Properties.VariableNames = cols;
                f{i_s} = ff_snsr.Freq;
                sigamp_mat{i_s} = ff_snsr.Amp;
                sigR_mat{i_s} = ff_snsr.Re;
                sigIm_mat{i_s} = ff_snsr.Im;
                sigcmplx_mat{i_s} =sigR_mat{i_s}+1i.*sigIm_mat{i_s};
            end
        end
        %%
        function [t,sig_tim]=...
                get_ff_tim(bf_nm_1,s_dir,...
                stn,date, time,n_s,ff_fldr,cols)
            t = cell(n_s, 1);
            fnm_ff = arrayfun(@(x) sprintf(bf_nm_1, x, stn, date, time),...
                s_dir, 'UniformOutput', false);
            cd ..

            fil_path = fullfile(ff_fldr, fnm_ff);

            ff_all = cellfun(@(x) readtable(x, 'ReadVariableNames',...
                false, 'HeaderLines', 5),...
                fil_path, 'UniformOutput', false);
            cd Matlab_codes
            for i_s = 1:n_s
                ff_snsr = ff_all{i_s};
                ff_snsr.Properties.VariableNames = cols;
                t{i_s} = ff_snsr.tim;
                sig_tim{i_s} = ff_snsr.val;
            end
        end
        %%
        function [f,sigamp_mat,sigR_mat,sigIm_mat,sigcmplx_mat]=...
                get_inpt_seisl(bf_nm_1,s_dir,...
                stn,datm,n_s,ff_fldr,cols)

            sigamp_mat = cell(n_s, 1);
            f = cell(n_s, 1);
            fnm_ff = arrayfun(@(x) sprintf(bf_nm_1, x, stn, datm),...
                s_dir, 'UniformOutput', false);
            cd ..

            fil_path = fullfile(ff_fldr, fnm_ff);
            ff_all = cellfun(@(x) readtable(x, 'ReadVariableNames',...
                false,'HeaderLines', 5),...
                fil_path, 'UniformOutput', false);
            cd Matlab_codes
            for i_s = 1:n_s
                ff_snsr = ff_all{i_s};
                ff_snsr.Properties.VariableNames = cols;
                f{i_s} = ff_snsr.Freq;
                sigamp_mat{i_s} = ff_snsr.Amp;
                sigR_mat{i_s} = ff_snsr.Re;
                sigIm_mat{i_s} = ff_snsr.Im;
                sigcmplx_mat{i_s} =sigR_mat{i_s}+1i.*sigIm_mat{i_s};
            end

        end
        %%
        function [bldamp_mat,bldcmplx_mat]=get_bldata_seisl(...
                bf_nm,stn,rec_vect,s_dir,ss_fldr,cols)
            for ir=1:length(rec_vect)
                rec=rec_vect(ir);
                fnm_ff = arrayfun(@(x)sprintf(bf_nm,x,stn,rec),...
                    s_dir, 'UniformOutput', false);
                cd ..

                fil_pth = fullfile(ss_fldr, fnm_ff);

                ff_all = cellfun(@(x)readtable(x,'ReadVariableNames',...
                    false, 'HeaderLines', 5),...
                    fil_pth, 'UniformOutput', false);
                cd Matlab_codes
                for i_s = 1:length(s_dir)
                    bld_snsr = ff_all{i_s};
                    bld_snsr.Properties.VariableNames = cols;
                    bldamp_mat{ir}(:,i_s) = bld_snsr.Amp;
                    bldR_mat{ir}(:,i_s) = bld_snsr.Re;
                    bldIm_mat{ir}(:,i_s)= bld_snsr.Im;
                    bldcmplx_mat{ir}(:,i_s) =bldR_mat{ir}(:,i_s)+...
                        1i.*bldIm_mat{ir}(:,i_s);
                end
            end
        end
        %%
        function [f,Uamp_mat,Ucpmlx_mat]=get_TF(n_str, n_rx, n_ry,...
                l_vect, b_vect, ftyp, V_s, L_f, B_f,...
                bf_nm,i_str,cmpt,n_c,rf_fldr,cols)
            for i_l=1:length(l_vect)
                l=l_vect(i_l);
                %                 for i_b=1:length(b_vect)
                b=b_vect(i_l);
                %                     if b>l
                %                         break
                %                     end

                fldr = fns_plot.get_fldrnm(n_str, n_rx, n_ry,...
                    l, b,ftyp, V_s, L_f, B_f);
                fil_nm = arrayfun(@(x) sprintf(bf_nm, x{1}, i_str,...
                    l, b),cmpt, 'UniformOutput', false);

                cd ..
                cd Results_Ansys
                fil_pth = fullfile(rf_fldr, fldr, fil_nm);
                U_all = cellfun(@(x) readtable(x), fil_pth,...
                    'UniformOutput', false);
                cd ..
                cd Matlab_codes

                for i_c = 1:n_c
                    Uc = U_all{i_c};
                    Uc.Properties.VariableNames = cols;
                    if i_l == 1
                        f = Uc.Freq;
                        Uamp_mat{i_c} = Uc.AMPL;
                        UR_mat{i_c} = Uc.REAL;
                        UIm_mat{i_c} = Uc.IMAG;
                        Ucpmlx_mat{i_c} = Uc.REAL+1i.*Uc.IMAG;
                    else
                        Uamp_mat{i_c}(:, i_l)= Uc.AMPL;
                        UR_mat{i_c}(:, i_l)= Uc.REAL;
                        UIm_mat{i_c}(:, i_l)= Uc.IMAG;
                        Ucpmlx_mat{i_c}(:, i_l)=Uc.REAL+1i.*Uc.IMAG;
                    end
                    %                     end
                end
            end
        end
        %%
        function [f,Uamp_mat,Ucpmlx_mat]=get_TFvSvary(n_str, n_rx, n_ry,...
                l_vect, b_vect, ftyp, V_s, L_f, B_f,...
                bf_nm,i_str,cmpt,n_c,rf_fldr,cols)
            for i_l=1:length(l_vect)
                l=l_vect(i_l);
                %                 for i_b=1:length(b_vect)
                b=b_vect(i_l);
                %                     if b>l
                %                         break
                %                     end

                fldr = fns_plot.get_fldrnm(n_str, n_rx, n_ry,...
                    l, b,ftyp, V_s, L_f, B_f);
                fil_nm = arrayfun(@(x) sprintf(bf_nm, x{1}, i_str,...
                    V_s),cmpt, 'UniformOutput', false);

                cd ..
                cd Results_Ansys
                fil_pth = fullfile(rf_fldr, fldr, fil_nm);
                U_all = cellfun(@(x) readtable(x), fil_pth,...
                    'UniformOutput', false);
                cd ..
                cd Matlab_codes

                for i_c = 1:n_c
                    Uc = U_all{i_c};
                    Uc.Properties.VariableNames = cols;
                    if i_l == 1
                        f = Uc.Freq;
                        Uamp_mat{i_c} = Uc.AMPL;
                        UR_mat{i_c} = Uc.REAL;
                        UIm_mat{i_c} = Uc.IMAG;
                        Ucpmlx_mat{i_c} = Uc.REAL+1i.*Uc.IMAG;
                    else
                        Uamp_mat{i_c}(:, i_l)= Uc.AMPL;
                        UR_mat{i_c}(:, i_l)= Uc.REAL;
                        UIm_mat{i_c}(:, i_l)= Uc.IMAG;
                        Ucpmlx_mat{i_c}(:, i_l)=Uc.REAL+1i.*Uc.IMAG;
                    end
                    %                     end
                end
            end
        end

        %%
        function [f,Uamp_mat,Ucpmlx_mat]=get_U_rec(n_str, n_rx, n_ry,...
                l_vect, b_vect, ftyp, V_s, L_f, B_f,rec,...
                bf_nm,i_str,cmpt,n_c,rf_fldr,cols)
            for i_l=1:length(l_vect)
                l=l_vect(i_l);
                for i_b=1:length(b_vect)
                    b=b_vect(i_b);
                    if b>l
                        break
                    end

                    fldr = fns_plot.get_fldrnm_rec(n_str, n_rx, n_ry,...
                        l, b,ftyp, V_s, L_f, B_f,rec);
                    fil_nm = arrayfun(@(x) sprintf(bf_nm, x{1}, i_str,...
                        l, b),cmpt, 'UniformOutput', false);

                    cd ..
                    fil_pth = fullfile(rf_fldr, fldr, fil_nm);
                    U_all = cellfun(@(x) readtable(x), fil_pth,...
                        'UniformOutput', false);
                    cd Matlab_codes

                    for i_c = 1:n_c
                        Uc = U_all{i_c};
                        Uc.Properties.VariableNames = cols;
                        if i_l == 1
                            f = Uc.Freq;
                            Uamp_mat{i_c} = Uc.AMPL;
                            UR_mat{i_c} = Uc.REAL;
                            UIm_mat{i_c} = Uc.IMAG;
                            Ucpmlx_mat{i_c} = Uc.REAL+1i.*Uc.IMAG;
                        else
                            Uamp_mat{i_c}(:, i_l)= Uc.AMPL;
                            UR_mat{i_c}(:, i_l)= Uc.REAL;
                            UIm_mat{i_c}(:, i_l)= Uc.IMAG;
                            Ucpmlx_mat{i_c}(:, i_l)=Uc.REAL+1i.*Uc.IMAG;
                        end
                    end
                end
            end
        end
    end
end
