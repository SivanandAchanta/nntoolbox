function [tot_err] = compute_error(data,targets,clv,numbats,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,ff,nl,cfn,dnn_flag,W,b,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn)

tot_err = 0;
gpu_flag = 0;

for li  = 1:numbats
    
    [X,Y,sl] = get_XY_seqver(data, targets, clv, (1:numbats), li, gpu_flag);   
    
    % Forward Pass
    if dnn_flag
        [ol] = fpav_cpu(X,W,b,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,sl);
        [otl] = get_otl(sl,nl_dnn,nlv_dnn);
        ol_mat = reshape(ol(1,otl(end-1):otl(end)-1),sl,nl_dnn(end));
        Xb = X;
        X = ol_mat;
    end
    [~,~,~,~,~,~,~,ym] = fp_lstm(X,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,nl,sl,ff,U,bu);
    
    % Cost Funtion
    switch cfn
        case 'nll'
            me = compute_zerooneloss(ym,Y);
        case 'ls'
            me = compute_nmlMSE(ym,Y);
    end
    
    tot_err     = tot_err + me/numbats;
    
end
