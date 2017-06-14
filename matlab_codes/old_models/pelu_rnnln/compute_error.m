function [total_err] = compute_error(data,targets,clv,numbats,gpu_flag,GWi,GWfr,GU,Gbh,Gbo,Gapelu,Gbpelu,Ggn,Gbn,f,nl,cfn,a_tanh,b_tanh)

total_err = 0;
% error computation
for li = 1:numbats
    [X,Y,sl] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);
    
    % fp
    if gpu_flag
        [pacm,bacm,acm,hcm,muv,sigv,ol_mat] = fp_gpu(X,GWi,GWfr,GU,Gbh,Gbo,Gapelu,Gbpelu,Ggn,Gbn,f,nl,a_tanh,b_tanh,sl);
    else
        [pacm,bacm,acm,hcm,muv,sigv,ol_mat] = fp_cpu(X,GWi,GWfr,GU,Gbh,Gbo,Gapelu,Gbpelu,Ggn,Gbn,f,nl,a_tanh,b_tanh,sl);
    end
    
    switch cfn
        case 'nll'
            me = compute_zerooneloss(ol_mat,Y);
        case 'ls'
            me = compute_nmlMSE(ol_mat,Y);
    end
    
    total_err = total_err + me/numbats;
    
end
