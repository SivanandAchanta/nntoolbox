function [J] = compute_Fofx(X,Y,GWi,GWfr,Gbh,GU,Gbo,f,nl,a_tanh,b_tanh,sl,cfn,tsteps,nng)

% Forward Pass
[scm,hcm,ol_mat] = fp_cpu(X,GWi,GWfr,GU,Gbh,Gbo,f,nl,a_tanh,b_tanh,sl,tsteps,nng);


% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ol_mat),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ol_mat)),2));
        
end


end