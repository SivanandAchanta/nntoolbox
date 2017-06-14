function [J] = compute_Fofx(X,Y,p_lf_1,p_lb_1,p_f_1,p_f_2,nl,sl,f,cfn)

% forward pass thru the model
fp_model

% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ym),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ym)),2));
        
end


end
