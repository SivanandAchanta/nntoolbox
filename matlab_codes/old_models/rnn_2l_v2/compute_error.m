function [total_err] = compute_error(data,targets,clv,numbats,gpu_flag,Gpi1,Gpi2,Gpo,f,nl,cfn,a_tanh,b_tanh)

total_err = 0;
% error computation
for li = 1:numbats
    [X,Y,sl] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);
    
    % fp
    fp_model

    switch cfn
        case 'nll'
            me = compute_zerooneloss(ym,Y);
        case 'ls'
            me = compute_nmlMSE(ym,Y);
    end
    
    total_err = total_err + me/numbats;
    
end
