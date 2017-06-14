function [total_err] = compute_error(data,targets,clv,numbats,gpu_flag,Gpi_1,Gp_rf_1,Gp_rb_1,Gp_rf_2,Gp_rb_2,Gp_rf_3,Gp_rb_3,Gp_rf_4,Gp_rb_4,Gpo_1,Gpo_2,f,nl,cfn)

total_err = 0;
% error computation
for li = 1:numbats
    [X,Y,sl] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);
    
    fp_model
    
    switch cfn
        case 'nll'
            me = compute_zerooneloss(ym,Y);
        case 'ls'
            me = compute_nmlMSE(ym,Y);
    end
    
    total_err = total_err + me/numbats;
    
end
