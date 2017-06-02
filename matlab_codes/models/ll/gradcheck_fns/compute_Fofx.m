function [J] = compute_Fofx(X,Y,Gpi1,Gpo,f,nl,sl,cfn,dm1)

% Forward Pass
hm1 = fp_cpu_ll(X,Gpi1,f(1));
hm1 = hm1.*dm1;
ol_mat = fp_cpu_ll(hm1,Gpo,f(end));

% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ol_mat),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ol_mat)),2));
        
end


end
