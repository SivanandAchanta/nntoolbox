function [J] = compute_Fofx(X,Y,GWi,GWfr,Gbh,GU,Gbo,Ggn,Gbn,f,nl,a_tanh,b_tanh,sl,cfn,lr_fw,dr_fw)

% Forward Pass
[scm,alpham,pacm,bacm,acm,hcm,muv,sigv,ol_mat] = fp_cpu(X,GWi,GWfr,GU,Gbh,Gbo,Ggn,Gbn,f,nl,a_tanh,b_tanh,sl,lr_fw,dr_fw);


% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ol_mat),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ol_mat)),2));
        
end


end