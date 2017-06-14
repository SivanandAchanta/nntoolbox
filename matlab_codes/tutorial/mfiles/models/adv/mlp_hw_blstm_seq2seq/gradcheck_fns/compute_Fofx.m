function [J] = compute_Fofx(X,Y,Yn,p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_0,p_f_3_1,p_f_3_2,p_f_1_dec,p_f_2_dec,p_lf_1_dec,p_lf_2_dec,p_lf_3_dec,p_f_4_1_dec,p_f_4_2_dec,nl,sl_enc,sl_dec,f,cfn,dm_f_1,dm_f_2,dm_f_1_dec,dm_f_2_dec)


fp_model_check

% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ym),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ym)),2));
        
end


end
