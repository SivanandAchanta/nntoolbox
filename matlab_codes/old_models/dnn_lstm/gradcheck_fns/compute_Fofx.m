function [J] = compute_Fofx(X,Y,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,nl,sl,f,cfn,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,dnn_flag)


% Forward Pass
if dnn_flag
    [ol] = fpav_cpu(X,W,b,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,sl);
    [otl] = get_otl(sl,nl_dnn,nlv_dnn);
    ol_mat = reshape(ol(1,otl(end-1):otl(end)-1),sl,nl_dnn(end));
    Xb = X;
    X = ol_mat;
end
[~,~,~,~,~,~,~,ym] = fp_lstm(X,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,nl,sl,f,U,bu);

% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ym),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ym)),2));
        
end


end