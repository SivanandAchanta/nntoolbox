function [J] = compute_Fofx(X,Y,GWi,GWfr,Gbh,GU,GUfr,Gbo,Ggn,Gbn,f,nl,a_tanh,b_tanh,sl,cfn,gt_flag)

% Forward Pass
if gt_flag
    [pacm,bacm,acm,hcm,muv,sigv,ol_mat] = fp_cpu_train(X,Y,GWi,GWfr,GU,GUfr,Gbh,Gbo,Ggn,Gbn,f,nl,a_tanh,b_tanh,sl);
else
    [pacm,bacm,acm,hcm,muv,sigv,ol_mat] = fp_cpu_test(X,GWi,GWfr,GU,GUfr,Gbh,Gbo,Ggn,Gbn,f,nl,a_tanh,b_tanh,sl);        
end


% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ol_mat),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ol_mat)),2));
        
end


end