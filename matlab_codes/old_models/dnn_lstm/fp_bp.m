% Forward Pass
if dnn_flag
    [ol] = fpav_cpu(X,W,b,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,sl);
    [otl] = get_otl(sl,nl_dnn,nlv_dnn);
    ol_mat = reshape(ol(1,otl(end-1):otl(end)-1),sl,nl_dnn(end));
    Xb = X;
    X = ol_mat;
end

[zm,im,fm,cm,om,hcm,hm,ym] = fp_lstm(X,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,nl,sl,f,U,bu);

% Error at Output Layer
switch cfn
    case 'ls'
        E   = -(Y - ym)/sl;
    case  'nll'
        E   = -(Y - ym)/sl;
end

% Backprop
gU = (E'*hm);
gbu = sum(E,1)';
Eb = U'*E';

[dhm,dom,dcm,dfm,dim,dzm] = bp_lstm(Eb,Rz,Ri,pi,Rf,pf,Ro,po,zm,im,fm,cm,om,hcm,nl,sl);
Ea = Wz'*dzm' + Wi'*dim' + Wf'*dfm' + Wo'*dom';
[gWz,gRz,gbz,gWi,gRi,gpi,gbi,gWf,gRf,gpf,gbf,gWo,gRo,gpo,gbo] = gradients_lstm(X,hm,cm,dom,dfm,dim,dzm,nl);

if dnn_flag
    [gW,gb] = bpav_cpu(Xb,Ea',ol,W,otl,btl_dnn,wtl_dnn,f_dnn,sl,nl_dnn,nh_dnn,l1,l2);
end
