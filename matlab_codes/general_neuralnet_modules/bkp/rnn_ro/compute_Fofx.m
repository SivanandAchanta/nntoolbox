function [J] = compute_Fofx(X,Y,GWi,GWfr,Gbh,GU,GUfr,Gbo,f,nl,a_tanh,b_tanh,sl,cfn)

% Forward Pass
[hcm,ol_mat] = fp_cpu(X,GWi,GWfr,GU,GUfr,Gbh,Gbo,f,nl,a_tanh,b_tanh,sl);


% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ol_mat),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ol_mat)),2));
        
end


end