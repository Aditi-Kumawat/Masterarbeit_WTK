%test
%cr = [0.4,6,0.2];
%a = fn_evaluate_A_criteria(5.0, 0.1,cr ,false)
function required_str = fn_evaluate_A_criteria(KB_f_max,KB_Ftr_,criteria,frequent_or_not)
    Au = criteria(1);
    Ao = criteria(2);
    Ar = criteria(3);
    if KB_f_max <= Au
        required_str = "Au passed";
    else
        if KB_f_max <= Ao
            if frequent_or_not
                if KB_Ftr_ <= Ar
                    required_str = "Ar passed";
                else
                    required_str = "failed";
                end
            else
                required_str = "Ao passed";
            end          
        else
            required_str = "failed";
        end
    end
end