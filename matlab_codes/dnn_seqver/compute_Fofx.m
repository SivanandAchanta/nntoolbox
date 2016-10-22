function [J] = compute_Fofx(X,Y,W,b,fl,nl,nh,nlv,wtl,btl,a_tanh,b_tanh,bs,cfn)


% Forward Pass
[ol] = fpav_cpu(X,W,b,nl,fl,nh,wtl,btl,a_tanh,b_tanh,bs);

[otl] = get_otl(bs,nl,nlv);
ol_mat = reshape(ol(1,otl(end-1):otl(end)-1),bs,nl(end));
    
% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ol_mat),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ol_mat)),2));
        
end


end