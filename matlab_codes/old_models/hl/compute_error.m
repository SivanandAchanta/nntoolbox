function [total_err] = compute_error(data,targets,clv,numbats,gpu_flag,Gpi1,Gpo,f,nl,cfn)

total_err = 0;
% error computation
for li = 1:numbats
    [X,Y,sl] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);
    
    [~,~,hm] = fp_hl(X,Gpi1);
    ol_mat = fp_cpu_ll(hm,Gpo,f(end));
    
    
    switch cfn
        case 'nll'
            me = compute_zerooneloss(ol_mat,Y);
        case 'ls'
            me = compute_nmlMSE(ol_mat,Y);
    end
    
    total_err = total_err + me/numbats;
    
end
