function [J] = compute_Fofx(X,Y,Gpi1,Gpi2,Gpo,f,nl,sl,cfn)

% Forward Pass
[tm1,htm1,hm1] = fp_hl(X,Gpi1);
[tm2,htm2,hm2] = fp_hl(hm1,Gpi2);
ol_mat = fp_cpu_ll(hm2,Gpo,f(end));

% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ol_mat),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ol_mat)),2));
        
end


end