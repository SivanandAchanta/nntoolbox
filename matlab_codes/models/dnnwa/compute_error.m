function [total_err] = compute_error(data,targets,clv,numbats,gpu_flag,GWi,Gbi,GWa,Gba,GWh,Gbh,GWo,Gbo,f,cfn)

total_err = 0;
% error computation
for li = 1:numbats
    [X,Y,bs] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);
    Y = Y(1,:);
    
    % fp
    [hcm,ah,alpha,c,sm,ol_mat] = fp_cpu(X,GWi,Gbi,GWa,Gba,GWh,Gbh,GWo,Gbo,f);
    
    switch cfn
        case 'nll'
            me = compute_zerooneloss(ol_mat,Y);
        case 'ls'
            me = compute_nmlMSE(ol_mat,Y);
    end
    
    total_err = total_err + me/numbats;
    
end


end
