function [total_err] = compute_error(data,targets,clv,numbats,gpu_flag,GW,Gb,nl,nlv,fl,nh,wtl,btl,cfn,a_tanh,b_tanh)

total_err = 0;
% error computation
for li = 1:numbats
    [X,Y,bs] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);
    
    % fp        
    if gpu_flag
        [ol] = fpav_gpu(X,GW,Gb,nl,fl,nh,wtl,btl,a_tanh,b_tanh,bs);
    else
        [ol] = fpav_cpu(X,GW,Gb,nl,fl,nh,wtl,btl,a_tanh,b_tanh,bs);
    end
    
    % compute error
    [otl] = get_otl(bs,nl,nlv);
    ol_mat = reshape(ol(1,otl(end-1):otl(end)-1),bs,nl(end));
    
    switch cfn
        case 'nll'
            me = compute_zerooneloss(ol_mat,Y);
        case 'ls'
            me = compute_nmlMSE(ol_mat,Y);
    end
    
    total_err = total_err + me/numbats;
    
end
